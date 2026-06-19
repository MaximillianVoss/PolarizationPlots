# PolarizationPlots

<!-- codex-repo-note:start -->
## Справка о репозитории / Repository note

**RU:** инструмент для построения графиков поляризации.

**EN:** a tool for plotting polarization data.

**Статус / Status:** активный проект 2026 года; ожидает рефакторинга и переименования. / active 2026 project; refactoring and repository rename are pending.

**Текущее имя / Current name:** `PolarizationPlots`

**Плановое имя / Planned name:** `polarization-plots`

**Topics:** `cleanup-pending`, `data-visualization`, `needs-rename`, `needs-review`, `physics`, `polarization`, `python`, `status-active`, `type-tool`
<!-- codex-repo-note:end -->

## Экспериментальный PySide6-прототип

Основное Tkinter-приложение запускается как раньше:

```powershell
python main.py
```

Отдельный прототип вкладки `Анализ r_min` на PySide6 запускается так:

```powershell
python -m pip install -r requirements-pyside6.txt
python main_pyside6_rmin.py
```

Прототип не заменяет текущий интерфейс. Он использует существующие расчёты траекторий,
метрики `r_min`, экспорт XLSX и отрисовку Matplotlib, но проверяет Qt-компоновку,
toolbar, карточки параметров, таблицу сводки и правую панель вывода.

