"""ETL: raw Yambda-50M -> flat numpy sequences, targets and item metadata.

Training histories keep every listen. Evaluation targets follow the official
protocol: a listen is relevant when `played_ratio_pct >= 50`.

The public Kaggle mirror ships the official train/val/test split. When only the
raw HF parquet is available we reproduce the official global temporal split
(train / 30 min gap / 1 day val / 30 min gap / 1 day test).
"""

from __future__ import annotations

import glob
import json
import os
import urllib.request

import duckdb
import numpy as np

from .utils import Timer, json_dump

ACTIVE_HORIZON = 7 * 86400


def _discover(cfg: dict) -> dict:
    root = cfg["data"]["input_root"]
    train = sorted(glob.glob(os.path.join(root, "**", "multi_event_train.parquet"), recursive=True))
    if train:
        d = os.path.dirname(train[0])
        return {
            "kind": "multi_event",
            "train": train[0],
            "val": os.path.join(d, "multi_event_val.parquet"),
            "test": os.path.join(d, "multi_event_test.parquet"),
        }
    listens = sorted(glob.glob(os.path.join(root, "**", "flat", "50m", "listens.parquet"), recursive=True))
    if listens:
        return {"kind": "flat", "listens": listens[0],
                "likes": listens[0].replace("listens.parquet", "likes.parquet")}
    raise FileNotFoundError(f"no Yambda-50M parquet under {root}")


def _download(cfg: dict) -> dict:
    raw = os.path.join(cfg["work_dir"], "raw")
    os.makedirs(raw, exist_ok=True)
    out = {}
    for name in ("listens", "likes"):
        path = os.path.join(raw, f"{name}.parquet")
        if not os.path.exists(path):
            url = f"{cfg['data']['hf_base']}/{name}.parquet"
            print("downloading", url, flush=True)
            urllib.request.urlretrieve(url, path + ".part")
            os.replace(path + ".part", path)
        out[name] = path
    return {"kind": "flat", **out}


def _columns(con, path: str) -> list[str]:
    return [r[0] for r in con.sql(f"DESCRIBE SELECT * FROM read_parquet('{path}')").fetchall()]


def _fetch(con, sql: str, n: int, dtypes: dict) -> dict:
    """Stream a big result into preallocated numpy arrays."""
    out = {k: np.empty(n, dtype=dt) for k, dt in dtypes.items()}
    i = 0
    for batch in con.execute(sql).fetch_record_batch(4_000_000):
        size = batch.num_rows
        for k in dtypes:
            out[k][i : i + size] = batch.column(batch.schema.get_field_index(k)).to_numpy(zero_copy_only=False)
        i += size
    assert i == n, (i, n)
    return out


def _bounds(uid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    change = np.flatnonzero(np.diff(uid)) + 1
    offsets = np.concatenate([[0], change, [len(uid)]]).astype(np.int64)
    return uid[offsets[:-1]], offsets


def _map_items(item: np.ndarray, vocab: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(vocab, item)
    bad = (idx >= len(vocab)) | (vocab[np.minimum(idx, len(vocab) - 1)] != item)
    if bad.any():
        raise ValueError(f"{int(bad.sum())} items outside the vocabulary")
    return idx.astype(np.int32)


def _load_seq(con, table: str, vocab: np.ndarray, liked: bool = True) -> dict:
    n = int(con.sql(f"SELECT count(*) FROM {table}").fetchone()[0])
    liked_expr = "liked" if liked else "FALSE AS liked"
    sql = (
        f"SELECT uid, item_id, timestamp, played_ratio_pct, is_organic, {liked_expr} FROM {table} "
        "ORDER BY uid, timestamp, item_id"
    )
    raw = _fetch(
        con, sql, n,
        {"uid": np.int64, "item_id": np.int64, "timestamp": np.int64,
         "played_ratio_pct": np.float64, "is_organic": np.uint8, "liked": np.uint8},
    )
    uids, offsets = _bounds(raw["uid"])
    ts = raw["timestamp"].astype(np.int32)
    dt = np.full(n, -1, dtype=np.int32)
    if n > 1:
        d = np.diff(ts)
        dt[:-1] = np.where(d > 0, d, 0)
    active = np.zeros(n, dtype=bool)
    for lo, hi in zip(offsets[:-1], offsets[1:]):
        if hi - lo < 2:
            continue
        seg = ts[lo:hi]
        nxt = np.searchsorted(seg, seg + ACTIVE_HORIZON, side="right")
        active[lo:hi] = nxt > np.arange(1, hi - lo + 1)
        dt[hi - 1] = -1
    return {
        "uids": uids,
        "offsets": offsets,
        "items": _map_items(raw["item_id"], vocab),
        "ts": ts,
        "ratio": np.clip(raw["played_ratio_pct"], 0, 255).astype(np.uint8),
        "organic": raw["is_organic"].astype(bool),
        "liked": raw["liked"].astype(bool),
        "dt_next": dt,
        "active7": active,
    }


def _save_seq(path: str, seq: dict) -> dict:
    np.savez(path, **seq)
    lengths = np.diff(seq["offsets"])
    return {
        "users": int(len(seq["uids"])),
        "events": int(len(seq["items"])),
        "median_len": float(np.median(lengths)) if len(lengths) else 0.0,
        "p90_len": float(np.percentile(lengths, 90)) if len(lengths) else 0.0,
    }


def _subset_targets(seq: dict, min_ratio: int, keep_uids: np.ndarray) -> dict:
    lens = np.diff(seq["offsets"])
    uid_ev = np.repeat(seq["uids"], lens)
    mask = (seq["ratio"] >= min_ratio) & np.isin(uid_ev, keep_uids)
    uids, offsets = _bounds(uid_ev[mask])
    return {"uids": uids, "offsets": offsets, "items": seq["items"][mask].astype(np.int32)}


def run(cfg: dict, log, source: str = "auto") -> dict:
    os.makedirs(cfg["data_dir"], exist_ok=True)
    con = duckdb.connect()
    con.execute("PRAGMA threads=4")
    con.execute(f"PRAGMA memory_limit='{cfg.get('duckdb_memory', '12GB')}'")
    con.execute(f"PRAGMA temp_directory='{os.path.join(cfg['work_dir'], 'duckdb_tmp')}'")
    con.execute("PRAGMA preserve_insertion_order=false")

    if source == "hf":
        paths = _download(cfg)
    else:
        try:
            paths = _discover(cfg)
        except FileNotFoundError:
            if source == "kaggle":
                raise
            log.warning("attached dataset not found, falling back to HF download")
            paths = _download(cfg)
    log.info("source: %s", paths)

    stats: dict = {"source": paths["kind"], "splits": {}}
    with Timer(log, "load + split"):
        if paths["kind"] == "multi_event":
            cols = _columns(con, paths["train"])
            if "event_type" not in cols:
                raise ValueError("multi_event file has no event_type column")
            tcol = "track_length_seconds" if "track_length_seconds" in cols else "track_length"
            for name in ("train", "val", "test"):
                con.execute(f"CREATE VIEW {name}_raw AS SELECT * FROM read_parquet('{paths[name]}')")
            ev = con.sql("SELECT event_type, count(*) n FROM train_raw GROUP BY 1 ORDER BY 2 DESC").fetchall()
            log.info("event types: %s", ev)
        else:
            tcol = "track_length_seconds" if "track_length_seconds" in _columns(con, paths["listens"]) else "track_length"
            con.execute(f"CREATE VIEW listens AS SELECT *, 'listen' AS event_type FROM read_parquet('{paths['listens']}')")
            con.execute(f"CREATE VIEW likes AS SELECT *, 'like' AS event_type FROM read_parquet('{paths['likes']}')")
            con.execute("CREATE VIEW raw AS SELECT * FROM listens UNION ALL BY NAME SELECT * FROM likes")
            t = con.sql("SELECT min(timestamp), max(timestamp) FROM raw").fetchone()
            test_ts = int(cfg["data"]["last_timestamp"]) - int(cfg["data"]["test_size"])
            val_ts = test_ts - int(cfg["data"]["gap_size"]) - int(cfg["data"]["val_size"])
            train_end = val_ts - int(cfg["data"]["gap_size"])
            con.execute(f"CREATE VIEW train_raw AS SELECT * FROM raw WHERE timestamp < {train_end}")
            con.execute(f"CREATE VIEW val_raw AS SELECT * FROM raw WHERE timestamp >= {val_ts} AND timestamp < {test_ts - int(cfg['data']['gap_size'])}")
            con.execute(f"CREATE VIEW test_raw AS SELECT * FROM raw WHERE timestamp >= {test_ts}")
            stats["split_bounds"] = {"train_end": train_end, "val": val_ts, "test": test_ts}
            log.info("flat split bounds: %s (data %s..%s)", stats["split_bounds"], t[0], t[1])

    for name in ("train", "val", "test"):
        con.execute(
            f"CREATE VIEW {name}_listens AS SELECT uid, item_id, timestamp, played_ratio_pct, "
            f"is_organic, {tcol} AS track_length FROM {name}_raw WHERE event_type = 'listen'"
        )
        row = con.sql(f"SELECT count(*), count(DISTINCT uid), count(DISTINCT item_id), min(timestamp), max(timestamp) "
                      f"FROM {name}_listens").fetchone()
        stats["splits"][name] = {"listens": int(row[0]), "users": int(row[1]), "items": int(row[2]),
                                 "min_ts": int(row[3]), "max_ts": int(row[4])}
        log.info("%s listens: %s", name, row)

    with Timer(log, "vocabulary"):
        con.execute(
            "CREATE TABLE vocab AS SELECT item_id, count(*) n FROM ("
            "SELECT item_id FROM train_listens UNION ALL "
            "SELECT item_id FROM val_listens UNION ALL "
            "SELECT item_id FROM test_listens) GROUP BY item_id"
        )
        vocab = con.sql("SELECT item_id FROM vocab ORDER BY item_id").fetchnumpy()["item_id"].astype(np.int64)
        stats["vocab_size"] = int(len(vocab))
        log.info("vocab: %d", len(vocab))

    with Timer(log, "likes"):
        con.execute("CREATE TABLE liked_items AS SELECT DISTINCT uid, item_id "
                    "FROM train_raw WHERE event_type = 'like'")
        con.execute("CREATE VIEW train_events AS SELECT t.*, (l.item_id IS NOT NULL) AS liked "
                    "FROM train_listens t LEFT JOIN liked_items l USING (uid, item_id)")
        for name in ("val", "test"):
            con.execute(
                f"CREATE VIEW {name}_events AS SELECT t.*, (l.item_id IS NOT NULL) AS liked "
                f"FROM {name}_listens t LEFT JOIN liked_items l USING (uid, item_id)"
            )

    with Timer(log, "sequences"):
        seqs = {}
        for name in ("train", "val", "test"):
            seqs[name] = _load_seq(con, f"{name}_events", vocab)
            stats["splits"][name].update(_save_seq(os.path.join(cfg["data_dir"], f"{name}_seq.npz"), seqs[name]))
            log.info("%s sequence: %s", name, stats["splits"][name])

    train_uids = seqs["train"]["uids"]
    with Timer(log, "targets"):
        for name in ("val", "test"):
            tgt = _subset_targets(seqs[name], int(cfg["data"]["min_played_ratio"]), train_uids)
            np.savez(os.path.join(cfg["data_dir"], f"{name}_targets.npz"), **tgt)
            stats["splits"][name]["targets"] = int(len(tgt["items"]))
            stats["splits"][name]["target_users"] = int(len(tgt["uids"]))
            log.info("%s targets: %d events / %d users", name, len(tgt["items"]), len(tgt["uids"]))

    with Timer(log, "test history"):
        con.execute(
            "CREATE VIEW history_events AS SELECT uid, item_id, timestamp, played_ratio_pct, is_organic "
            "FROM train_listens UNION ALL SELECT uid, item_id, timestamp, played_ratio_pct, is_organic "
            "FROM val_listens"
        )
        hist = _load_seq(con, "history_events", vocab, liked=False)
        uid_ev = np.repeat(hist["uids"], np.diff(hist["offsets"]))
        keep_ev = np.isin(uid_ev, train_uids)
        uids, offsets = _bounds(uid_ev[keep_ev])
        items = hist["items"][keep_ev]
        np.savez(os.path.join(cfg["data_dir"], "test_history.npz"), uids=uids, offsets=offsets, items=items)
        stats["test_history"] = {"users": int(len(uids)), "events": int(len(items))}
        log.info("test history: %s", stats["test_history"])

    with Timer(log, "item metadata"):
        df = con.sql(
            "SELECT item_id, count(*) n, avg(played_ratio_pct) mean_ratio, "
            "avg(track_length) mean_length, avg(is_organic::INT) organic_share "
            "FROM train_listens GROUP BY item_id"
        ).arrow().to_pandas()
        counts = np.zeros(len(vocab), dtype=np.int32)
        mean_ratio = np.full(len(vocab), np.nan, dtype=np.float32)
        mean_length = np.full(len(vocab), np.nan, dtype=np.float32)
        organic_share = np.zeros(len(vocab), dtype=np.float32)
        idx = np.searchsorted(vocab, df["item_id"].to_numpy())
        counts[idx] = df["n"].to_numpy()
        mean_ratio[idx] = df["mean_ratio"].to_numpy()
        mean_length[idx] = df["mean_length"].to_numpy()
        organic_share[idx] = df["organic_share"].to_numpy()
        like_counts = con.sql("SELECT item_id, count(*) n FROM train_events WHERE liked GROUP BY item_id").arrow().to_pandas()
        like_rate = np.zeros(len(vocab), dtype=np.float32)
        if len(like_counts):
            li = np.searchsorted(vocab, like_counts["item_id"].to_numpy())
            like_rate[li] = like_counts["n"].to_numpy() / np.maximum(counts[li], 1)
        np.savez(
            os.path.join(cfg["data_dir"], "items.npz"),
            item_ids=vocab.astype(np.int32), counts=counts, mean_ratio=mean_ratio,
            mean_length=mean_length, organic_share=organic_share, like_rate=like_rate,
        )
        stats["items_without_train_listens"] = int((counts == 0).sum())
        stats["items_without_train_listens_share"] = float((counts == 0).mean())

    json_dump(stats, os.path.join(cfg["data_dir"], "stats.json"))
    log.info("stats: %s", json.dumps(stats["splits"], ensure_ascii=False))
    return stats
