# Yambda-50M: GPT-2 recommender, reward model, GRPO

Исследование влияния RL-дообучения генеративного рекомендателя на релевантность,
вовлечённость и поведение выдачи. Датасет — [Yambda-50M](https://huggingface.co/datasets/yandex/yambda)
(10 000 пользователей, 934 057 треков, 46.5M прослушиваний, 881k лайков).

Отчёт, результаты и команды воспроизведения — в конце работы. Пока репозиторий
содержит код пайплайна.

## Структура

```
run.py                 единая точка входа, стадии etl/eda/baseline/train/eval/rm/rl/analyze/all
src/yr/                код пайплайна
configs/default.yaml   гиперпараметры
scripts/kaggle_run.sh  запуск стадии на Kaggle GPU и скачивание артефактов
```

## Запуск на Kaggle

```bash
scripts/kaggle_run.sh etl        # подготовка данных + EDA
scripts/kaggle_run.sh baseline   # popularity baseline
scripts/kaggle_run.sh train      # GPT-2
scripts/kaggle_run.sh rm         # reward model
scripts/kaggle_run.sh rl         # GRPO / DPO / filtered BC
scripts/kaggle_run.sh all        # всё подряд в одном ядре (<= 12 ч)
```

Каждая стадия — отдельное Kaggle-ядро: код клонируется из этого репозитория,
сырые данные монтируются из публичного датасета
`dinaryakupov/yambda-50m-compressed` (официальный time-split Yambda-50M).

## Протокол

- test: последний день, val: предыдущий день, gap 30 минут — как в официальном
  benchmark Yambda.
- Прослушивание засчитывается при `played_ratio_pct >= 50`.
- Ранжирование по всему каталогу, NDCG/Recall/Coverage@10/100.
- Reward model: `played_ratio`, `like`, continuation/return-прокси; ensemble из 3 моделей.
- RL: GRPO по item-токенам, KL к reference, пессимистичная награда `mean - λ·std`.
