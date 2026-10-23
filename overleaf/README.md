# Отчёт для Overleaf

Компилировать **pdfLaTeX**, файл `main.tex`, библиография — BibTeX (`refs.bib`).

## Как загрузить

1. Overleaf → *New Project* → *Upload Project* → заархивировать эту папку целиком
   (`zip -r report.zip overleaf/`), либо *Blank Project* и загрузить файлы с сохранением структуры.
2. Порядок сборки: pdfLaTeX → BibTeX → pdfLaTeX ×2 (Overleaf делает это сам).

## Состав

| Файл | Назначение |
|---|---|
| `main.tex` | преамбула, порядок разделов |
| `yambda.sty` | стиль: шрифты T2A, поля, палитра, врезка `keybox`, подписи |
| `sections/00_abstract.tex` … `07_repro.tex` | текст по разделам |
| `refs.bib` | 6 источников (Yambda, GPT-2, SASRec, GRPO, DPO, Gao et al.) |
| `figures/fig_*.pdf` | 7 рисунков, вектор, 300 dpi-версии в `.png` |

## Проверки без TeX

```bash
python3 scripts/check_paper.py
```

Проверяет баланс скобок и окружений, все `\ref` против `\label`, наличие
файлов рисунков, `\cite` против `refs.bib`, посторонние символы.
Локального TeX нет, поэтому полноценная сборка не проверялась — при первой
сборке на Overleaf возможны только косметические замечания.

## Если сборка падает

- `Font T2A/cmr/... not loadable` → в `yambda.sty` уже подключён `lmodern`;
  проверьте, что он не удалён.
- `LaTeX Error: File 'figures/...' not found` → нарушена структура папок при загрузке.
- `Citation undefined` → первый прогон всегда даёт это; нужен BibTeX и ещё два прохода.

## Пересборка рисунков

```bash
uv venv .venv-figures && uv pip install --python .venv-figures/bin/python matplotlib seaborn
.venv-figures/bin/python scripts/make_paper_figures.py
```

Скрипт читает `results/kaggle/all/fresh/artifacts/` (свежее скачивание ядра)
и пишет PDF+PNG в `overleaf/figures/`.
