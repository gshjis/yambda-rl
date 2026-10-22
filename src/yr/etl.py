"""ETL: raw Yambda-50M -> flat numpy sequences, targets and item metadata.

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

LISTEN = "listen"


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
        likes = listens[0].replace("listens.parquet", "likes.parquet")
        return {"kind": "flat", "listens": listens[0], "likes": likes}
    raise FileNotFoundError(
        f"no Yambda-50M parquet under {root}; attach dinaryakupov/yambda-50m-compressed "
        "or run with --source hf"
    )


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


def _where(event_col: bool, batch) -> str:
    parts = []
    if event_col:
        parts.append("event_type = 'listen'")
    parts.append(f"played_ratio_pct >= {int(batch['data']['min_played_ratio'])}")
    return " WHERE " + " AND ".join(parts)


def _series(con, sql: str) -> dict:
    """Run an aggregate query and flatten list columns into numpy arrays."""
    tbl = con.sql(sql).arrow()
    cols = [c for c in tbl.column_names if c != "uid"]
    out = {c: [] for c in cols}
    uids = []
    for batch in tbl.to_batches(max_chunksize=256):
        for row in batch.to_pylist():
            uids.append(int(row["uid"]))
            for c in cols:
                out[c].append(row[c])
    return {"uids": np.asarray(uids, dtype=np.int64), **out}


def _flatten(rows: list[list], dtype) -> tuple[np.ndarray, np.ndarray]:
    lengths = np.fromiter((len(x) for x in rows), dtype=np.int64, count=len(rows))
    offsets = np.zeros(len(rows) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    flat = np.fromiter((v for x in rows for v in x), dtype=dtype, count=int(offsets[-1]))
    return flat, offsets


def _map_ids(rows: list[list], vocab_sorted: np.ndarray) -> list[np.ndarray]:
    """Map raw item ids to contiguous indices via binary search."""
    mapped = []
    for x in rows:
        arr = np.asarray(x, dtype=np.int64)
        idx = np.searchsorted(vocab_sorted, arr)
        if idx.size and (idx.max() >= len(vocab_sorted) or not np.array_equal(vocab_sorted[idx], arr)):
            raise ValueError("item outside vocabulary")
        mapped.append(idx.astype(np.int32))
    return mapped


def _event_labels(ts: np.ndarray, active_days: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """dt to the next listen (capped) and whether the user listens again within 7 days."""
    dt = np.full(len(ts), -1, dtype=np.int32)
    if len(ts) > 1:
        dt[:-1] = np.diff(ts)
    horizon = active_days * 86400
    # last event of the user has no future, so no continuation and no return
    active = np.zeros(len(ts), dtype=bool)
    if len(ts) > 1:
        end = np.searchsorted(ts, ts[:-1] + horizon, side="right")
        active[:-1] = end > np.arange(1, len(ts))
    return dt, active


def _save_sequences(path: str, user_rows: dict, vocab_sorted: np.ndarray, batch: dict) -> dict:
    uids = user_rows["uids"]
    items = _map_ids(user_rows["items"], vocab_sorted)
    ts_rows = user_rows["ts"]
    ratio_rows = user_rows["ratio"]
    organic_rows = user_rows["organic"]
    liked_rows = user_rows["liked"]

    items_flat, offsets = _flatten(items, np.int32)
    ts_flat = np.fromiter((v for x in ts_rows for v in x), dtype=np.int32, count=len(items_flat))
    ratio_flat = np.fromiter((int(v) for x in ratio_rows for v in x), dtype=np.uint8, count=len(items_flat))
    organic_flat = np.fromiter((int(v) for x in organic_rows for v in x), dtype=np.uint8,
                               count=len(items_flat)).astype(bool)
    liked_flat = np.fromiter((int(v) for x in liked_rows for v in x), dtype=np.uint8,
                             count=len(items_flat)).astype(bool)

    dt_flat = np.zeros(len(items_flat), dtype=np.int32)
    active_flat = np.zeros(len(items_flat), dtype=bool)
    for i in range(len(uids)):
        lo, hi = offsets[i], offsets[i + 1]
        dt_flat[lo:hi], active_flat[lo:hi] = _event_labels(ts_flat[lo:hi])

    np.savez(
        path,
        uids=uids,
        offsets=offsets,
        items=items_flat,
        ts=ts_flat,
        ratio=ratio_flat,
        organic=organic_flat,
        liked=liked_flat,
        dt_next=dt_flat,
        active7=active_flat,
    )
    return {
        "users": int(len(uids)),
        "events": int(len(items_flat)),
        "median_len": float(np.median(np.diff(offsets))),
        "p90_len": float(np.percentile(np.diff(offsets), 90)),
    }


def _targets_from_seq(path: str, seq: dict, keep_uids: set[int]) -> dict:
    keep = [i for i, u in enumerate(seq["uids"]) if int(u) in keep_uids]
    items = [seq["items"][i] for i in keep]
    flat, offsets = _flatten(items, np.int32)
    np.savez(path, uids=seq["uids"][keep], offsets=offsets, items=flat)
    return {
        "users": int(len(keep)),
        "events": int(len(flat)),
        "targets_per_user": float(np.mean(np.diff(offsets))) if len(keep) else 0.0,
    }


def run(cfg: dict, log, source: str = "auto") -> dict:
    os.makedirs(cfg["data_dir"], exist_ok=True)
    con = duckdb.connect()
    con.execute("PRAGMA threads=4")
    con.execute("PRAGMA memory_limit='6GB'")

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
            has_event = "event_type" in cols
            log.info("columns: %s, event_type=%s", cols, has_event)
            con.execute(f"CREATE VIEW train_raw AS SELECT * FROM read_parquet('{paths['train']}')")
            con.execute(f"CREATE VIEW val_raw AS SELECT * FROM read_parquet('{paths['val']}')")
            con.execute(f"CREATE VIEW test_raw AS SELECT * FROM read_parquet('{paths['test']}')")
            if not has_event:
                raise ValueError("multi_event file has no event_type column")
            ev = con.sql("SELECT event_type, count(*) n FROM train_raw GROUP BY 1 ORDER BY 2 DESC").fetchall()
            log.info("event types: %s", ev)
            tcol = "track_length_seconds" if "track_length_seconds" in cols else "track_length"
            for name in ("train", "val", "test"):
                t = con.sql(f"SELECT min(timestamp), max(timestamp), count(*) FROM {name}_raw").fetchone()
                stats["splits"][name + "_raw"] = {"min_ts": int(t[0]), "max_ts": int(t[1]), "rows": int(t[2])}
        else:
            has_event = False
            lcols = _columns(con, paths["listens"])
            tcol = "track_length_seconds" if "track_length_seconds" in lcols else "track_length"
            con.execute(
                f"CREATE VIEW listens AS SELECT *, 'listen' AS event_type FROM read_parquet('{paths['listens']}')"
            )
            con.execute(
                f"CREATE VIEW likes AS SELECT *, 'like' AS event_type FROM read_parquet('{paths['likes']}')"
            )
            con.execute("CREATE VIEW raw AS SELECT * FROM listens UNION ALL BY NAME SELECT * FROM likes")
            t = con.sql("SELECT min(timestamp), max(timestamp), count(*) FROM raw").fetchone()
            log.info("raw timestamps: %s .. %s (%s rows)", t[0], t[1], t[2])
            test_ts = int(cfg["data"]["last_timestamp"]) - int(cfg["data"]["test_size"])
            val_ts = test_ts - int(cfg["data"]["gap_size"]) - int(cfg["data"]["val_size"])
            train_end = val_ts - int(cfg["data"]["gap_size"])
            if not (train_end < val_ts < test_ts <= int(t[1]) + 1):
                raise ValueError(f"official split does not fit data range: {t[0]}..{t[1]}")
            con.execute(
                f"CREATE VIEW train_raw AS SELECT * FROM raw WHERE timestamp < {train_end}"
            )
            con.execute(
                f"CREATE VIEW val_raw AS SELECT * FROM raw WHERE timestamp >= {val_ts} AND timestamp < {test_ts - int(cfg['data']['gap_size'])}"
            )
            con.execute(f"CREATE VIEW test_raw AS SELECT * FROM raw WHERE timestamp >= {test_ts}")
            stats["split_bounds"] = {"train_end": train_end, "val": val_ts, "test": test_ts}
            has_event = True

    where = _where(has_event, cfg)
    with Timer(log, "listen aggregation"):
        for name in ("train", "val", "test"):
            con.execute(
                f"CREATE VIEW {name}_listens AS SELECT uid, item_id, timestamp, played_ratio_pct, "
                f"is_organic, {tcol} AS track_length FROM {name}_raw{where}"
            )
            n = con.sql(f"SELECT count(*) FROM {name}_listens").fetchone()[0]
            stats["splits"][name] = {"listens": int(n)}
            log.info("%s listens: %s", name, n)

    with Timer(log, "vocabulary"):
        con.execute(
            "CREATE TABLE vocab AS SELECT item_id, count(*) n FROM ("
            "SELECT item_id FROM train_listens UNION ALL "
            "SELECT item_id FROM val_listens UNION ALL "
            "SELECT item_id FROM test_listens) GROUP BY item_id"
        )
        vocab = con.sql("SELECT item_id FROM vocab ORDER BY item_id").fetchnumpy()["item_id"].astype(np.int64)
        stats["vocab_size"] = int(len(vocab))

    with Timer(log, "likes"):
        if has_event:
            con.execute(
                "CREATE TABLE liked_items AS SELECT DISTINCT uid, item_id FROM train_raw WHERE event_type = 'like'"
            )
        else:
            con.execute("CREATE TABLE liked_items AS SELECT DISTINCT uid, item_id FROM likes")
        con.execute(
            "CREATE VIEW train_events AS SELECT t.*, (l.item_id IS NOT NULL) AS liked "
            "FROM train_listens t LEFT JOIN liked_items l USING (uid, item_id)"
        )

    seq_sql = (
        "SELECT uid, list(item_id ORDER BY timestamp, item_id) items, "
        "list(timestamp ORDER BY timestamp, item_id) ts, "
        "list(played_ratio_pct ORDER BY timestamp, item_id) ratio, "
        "list(is_organic ORDER BY timestamp, item_id) organic, "
        "list(liked ORDER BY timestamp, item_id) liked "
    )
    seqs = {}
    for name in ("train", "val", "test"):
        with Timer(log, f"{name} sequences"):
            if name == "train":
                inner = "train_events t"
            else:
                inner = (
                    f"(SELECT t.*, (l.item_id IS NOT NULL) AS liked FROM {name}_listens t "
                    "LEFT JOIN liked_items l USING (uid, item_id)) t"
                )
            rows = _series(con, seq_sql + f"FROM {inner} GROUP BY uid ORDER BY uid")
            seqs[name] = {
                "uids": rows["uids"],
                "items": _map_ids(rows["items"], vocab),
            }
            stats["splits"][name].update(_save_sequences(
                os.path.join(cfg["data_dir"], f"{name}_seq.npz"), rows, vocab, cfg
            ))
    train_uids = set(seqs["train"]["uids"].tolist())

    with Timer(log, "val/test targets"):
        for name in ("val", "test"):
            stats["splits"][name].update(_targets_from_seq(
                os.path.join(cfg["data_dir"], f"{name}_targets.npz"), seqs[name], train_uids
            ))

    with Timer(log, "test history"):
        hist_sql = (
            "SELECT uid, list(item_id ORDER BY timestamp, item_id) items FROM ("
            "SELECT uid, item_id, timestamp FROM train_listens UNION ALL "
            "SELECT uid, item_id, timestamp FROM val_listens) GROUP BY uid ORDER BY uid"
        )
        rows_h = _series(con, hist_sql)
        keep = np.array([i for i, u in enumerate(rows_h["uids"]) if int(u) in train_uids], dtype=np.int64)
        hist_items = _map_ids([rows_h["items"][i] for i in keep], vocab)
        flat, offsets = _flatten(hist_items, np.int32)
        np.savez(
            os.path.join(cfg["data_dir"], "test_history.npz"),
            uids=rows_h["uids"][keep],
            offsets=offsets,
            items=flat,
        )
        stats["test_history"] = {"users": int(len(keep)), "events": int(len(flat))}

    with Timer(log, "item metadata"):
        meta = con.sql(
            "SELECT item_id, count(*) n, avg(played_ratio_pct) mean_ratio, "
            "avg(track_length) mean_length, avg(is_organic::INT) organic_share "
            "FROM train_listens GROUP BY item_id"
        ).arrow()
        df = meta.to_pandas()
        counts = np.zeros(len(vocab), dtype=np.int32)
        mean_ratio = np.full(len(vocab), np.nan, dtype=np.float32)
        mean_length = np.full(len(vocab), np.nan, dtype=np.float32)
        organic_share = np.zeros(len(vocab), dtype=np.float32)
        idx = np.searchsorted(vocab, df["item_id"].to_numpy())
        counts[idx] = df["n"].to_numpy()
        mean_ratio[idx] = df["mean_ratio"].to_numpy()
        mean_length[idx] = df["mean_length"].to_numpy()
        organic_share[idx] = df["organic_share"].to_numpy()
        like_counts = con.sql(
            "SELECT item_id, count(*) n FROM train_events WHERE liked GROUP BY item_id"
        ).arrow().to_pandas()
        like_rate = np.zeros(len(vocab), dtype=np.float32)
        if len(like_counts):
            idx_l = np.searchsorted(vocab, like_counts["item_id"].to_numpy())
            like_rate[idx_l] = like_counts["n"].to_numpy() / np.maximum(counts[idx_l], 1)
        np.savez(
            os.path.join(cfg["data_dir"], "items.npz"),
            item_ids=vocab.astype(np.int32),
            counts=counts,
            mean_ratio=mean_ratio,
            mean_length=mean_length,
            organic_share=organic_share,
            like_rate=like_rate,
        )
        stats["items_without_train_listens"] = int((counts == 0).sum())
        stats["items_without_train_listens_share"] = float((counts == 0).mean())

    json_dump(stats, os.path.join(cfg["data_dir"], "stats.json"))
    log.info("stats: %s", json.dumps(stats["splits"], ensure_ascii=False))
    return stats
