# Сводка по литературе и экспериментам

Актуально на 9 октября 2026 года.

## 1. Постановка работы

Работа исследует uncertainty structured output в многошаговых LLM-агентах.
Центральная проблема состоит не в том, умеет ли модель синтаксически вывести
JSON, а в том, можно ли до выполнения распознать формально корректное, но
семантически неверное действие (`wrong-valid`), которое изменит состояние
среды и вызовет ошибку всей последующей траектории.

Рабочая причинная цепочка:

`hard constrained decoding -> wrong-valid tool call -> state corruption ->
trajectory failure -> selective intervention`.

Structured output остаётся центральной темой. Современные модели обычно умеют
генерировать корректный формат, но гарантированный grammar/JSON decoding меняет
распределение следующего токена. Поэтому синтаксическая валидность не устраняет
риск, а может сделать ошибку менее заметной: вместо неисполняемого ответа агент
получает допустимый вызов с неправильной функцией или аргументами.

## 2. Что показывает reading pack

Reading pack содержит 22 работы. Они делятся на четыре содержательных блока.

### 2.1. Structured и constrained generation

- **Grammar-Aligned Decoding (15)** показывает, что простое grammar masking
  искажает исходное распределение модели и не обязательно воспроизводит
  корректное условное распределение при заданной грамматике.
- **The Hidden Cost of Structured Generation / DCCD (14)** формулирует
  projection tax: если допустимым продолжениям исходно назначена малая
  вероятность, перенормировка может направить генерацию в локально допустимый,
  но семантически неверный результат. Предлагаемый DCCD отделяет свободный
  semantic draft от последующего структурного enforcement.
- **JSONSchemaBench (18)** систематизирует проверку constrained decoding по
  покрытию JSON Schema, скорости и качеству итоговых ответов.
- **CONSTRUCT (19)** оценивает надёжность всего structured output и отдельных
  полей без доступа к logprobs. Это важный black-box baseline, но он не
  моделирует распространение ошибки после выполнения tool call.

Этот блок даёт теоретическое объяснение нашей метрики constraint pressure и
наблюдаемого constraint tax.

### 2.2. Uncertainty отдельного function call

- **UQ for LLM Function-Calling (13)** — наиболее прямой baseline: сравнивает
  single-sample logit uncertainty и multi-sample uncertainty, использует
  semantic-token scoring и AST-clustering вызовов.
- **SAGE (01)** оценивает uncertainty выбора инструмента и аргументов и решает,
  выполнить действие или задать уточняющий вопрос.
- **Confidence Dichotomy (09)** показывает, что разные типы инструментов могут
  по-разному влиять на calibration и overconfidence.
- **Action--Belief Gap (20)** демонстрирует, что заявленная confidence модели
  может не согласовываться с её фактическим поведением в интерактивной среде.

Вывод этого блока: confidence ответа в статической QA-задаче нельзя напрямую
считать confidence действия. Для tool use нужно оценивать конкретную функцию,
аргументы и последствия выполнения.

### 2.3. Propagation по многошаговой траектории

- **SAUP (07)** агрегирует per-step uncertainty с situation-aware весами.
- **UProp (06)** разделяет intrinsic uncertainty текущего решения и extrinsic
  uncertainty, унаследованную от предыдущих решений.
- **HTC / Agentic Confidence Calibration (05)** обучает trajectory calibrator
  на 48 признаках динамики token/top-k confidence между шагами.
- **Beyond Single-Turn Confidence (02)** сравнивает способы агрегации
  uncertainty на BFCL-V4 и многошаговых задачах.
- **RUPA (03)** представляет историю агента как граф reasoning states,
  tool interactions и feedback среды и распространяет риск по связям графа.
- **Agent UQ survey и trajectory taxonomy (08, 11, 12)** задают общую
  терминологию, trajectory-level calibration и checkpoint metrics.

Этот блок показывает, что многошаговая UQ уже является самостоятельным
направлением. Поэтому новизну нельзя формулировать как «первую uncertainty
работу для агентов». Наша более узкая ниша — влияние hard structured constraints
на wrong-valid state transitions и их раннее обнаружение.

### 2.4. Решение о вмешательстве

- **ToolChain-CRC (04)** формулирует conformal risk control и anytime alarm
  для tool-use trajectories.
- **From Uncertainty to Action / Value of Steering (16)** показывает, что
  uncertainty может предсказывать общий неуспех, но не обязательно тот шаг, на
  котором вмешательство действительно поможет.
- **PROUR: Clarify or Verify (17)** маршрутизирует uncertainty между ACT,
  CLARIFY и VERIFY.
- **Uncertainty of Thoughts и Conformal Information Pursuit (21, 22)**
  рассматривают последовательный запрос дополнительной информации.

Практическая метрика должна поэтому оценивать не только AUROC detector'а, но и
риск среди реально исполняемых действий, coverage, число пропущенных критических
ошибок и пользу/вред вмешательства.

## 3. Позиционирование относительно литературы

Наиболее защищаемая формулировка вклада:

> Hard structured constraints can transform a retryable invalid action into a
> schema-valid but semantically wrong state transition. We evaluate whether
> this transition can be detected before execution, how its risk propagates
> through a stateful tool-use trajectory, and when an agent should abstain,
> clarify, verify, or proceed.

Отличие от соседних работ:

- от function-calling UQ — переход от изолированного вызова к состоянию и
  финальному исходу траектории;
- от SAUP/UProp/HTC/RUPA — structured constraint рассматривается как возможный
  источник wrong-valid ошибки, а не только как формат действия;
- от SAGE/PROUR/VoS — отдельно измеряются state corruption и critical errors,
  прошедшие selective execution;
- от исследований constrained decoding — оценивается не только качество
  JSON/output, но и последствия его исполнения агентом.

## 4. Проведённые эксперименты

### 4.1. Синтетический proof of concept

На Qwen2.5-0.5B было проверено 116 structured function-calling примеров;
26% ответов были wrong-valid.

| Detector | Random-split AUROC | Leave-One-Family-Out AUROC |
| --- | ---: | ---: |
| G-NLL-SMT | 0.52 | 0.49 |
| Logit baselines | 0.61 | 0.23 |
| Constraint pressure | 0.88 | 0.73 |
| Baselines + CP | 0.91 | 0.73 |

Первоначально constraint pressure выглядел сильным и переносился между
синтетическими tool families. Однако последующие реальные BFCL-эксперименты
показали, что этот результат нельзя обобщать: синтетический набор содержал
family-level confound.

### 4.2. Constraint tax

На тех же 116 примерах сравнивались одинаковые prompt-only и constrained arms.

| Режим | correct | wrong-valid | invalid |
| --- | ---: | ---: | ---: |
| Prompt-only | 0.48 | 0.17 | 0.34 |
| Constrained | 0.76 | 0.24 | 0.00 |

Из 40 исходно invalid ответов 70% были исправлены ограничением, а 30%
превратились в wrong-valid. Таким образом, constraints одновременно повышают
точность и переводят часть громких format errors в тихие исполняемые ошибки.

Эффект подтвердился на реальном BFCL с Qwen2.5-0.5B:

| Slice | Prompt-only wrong-valid | Constrained wrong-valid |
| --- | ---: | ---: |
| simple | 0.30 | 0.36 |
| live | 0.50 | 0.68 |

Во всех constrained arms schema validity стала равна 100%. Это поддерживает
гипотезу о constraint tax, но не означает, что constrained decoding в целом
вреден: итоговая accuracy также выросла.

### 4.3. Wrong-valid detection на реальном BFCL

Проведены clean-label эксперименты на BFCL `multiple` и `live_simple` для
Qwen2.5-0.5B и Qwen2.5-3B. Исправлена ранняя ошибка grammar, удалявшая optional
arguments; после исправления все результаты были пересчитаны.

| Detector | 0.5B multiple | 0.5B live | 3B multiple | 3B live |
| --- | ---: | ---: | ---: | ---: |
| G-NLL-SMT | 0.84 | 0.81 | 0.83 | 0.84 |
| Logit baselines | 0.83 | 0.79 | 0.82 | 0.85 |
| Constraint pressure | 0.66 | 0.70 | 0.69 | 0.77 |
| Baselines + CP | 0.81 | 0.80 | 0.75 | 0.82 |

Главный отрицательный результат: на реальном BFCL constraint pressure слабее
обычных log-prob признаков, а добавление CP ухудшает combined detector. Этот
вывод воспроизводится при увеличении модели с 0.5B до 3B и при cross-slice
distribution shift.

### 4.4. Multi-sample и hidden-state baselines

На BFCL multiple для Qwen2.5-0.5B были проверены более дорогие методы:

| Метод | AUROC | AUPRC | Risk@80% |
| --- | ---: | ---: | ---: |
| G-NLL-SMT | 0.833 | 0.789 | 0.357 |
| AST entropy, 8 samples | 0.695 | 0.654 | 0.371 |
| INSIDE EigenScore | 0.641 | 0.560 | 0.414 |
| INSIDE mean-pool ablation | 0.634 | 0.566 | 0.400 |

AST entropy и INSIDE дают реальный, но более слабый сигнал и не улучшают
combined logit detector. Это согласуется с работой UQ for LLM Function-Calling:
для function calls дорогой multi-sample UQ не обязательно превосходит простой
single-sample logit score.

### 4.5. Критические высокоуверенные ошибки

Критической считается wrong-valid ошибка, которую selective policy принимает
как низкорисковую и разрешает исполнить. Проверены coverage 20%, 50% и 80%, а
также буквальная token confidence 0.80/0.90/0.95.

При semantic-token confidence не ниже 0.95 были обнаружены:

- 17 wrong-valid среди 81 высокоуверенного вызова на 0.5B `multiple`;
- 37 wrong-valid среди 89 высокоуверенных вызовов на 0.5B `live_simple`.

Следовательно, высокая token confidence не гарантирует правильность действия.
На safest 20% G-NLL-SMT пропустил 1/35 ошибку на `multiple`, но 11/44 на более
трудном `live_simple`; CP-only пропустил соответственно 11/35 и 27/44.
Semantic-token прогон для 3B ещё не выполнен, поэтому его текущие literal
confidence результаты используют менее чистую all-token оценку.

### 4.6. Stateful BFCL multi-turn и HTC

Реализован rollout по официальному BFCL stateful executor: после каждого
constrained action выполняется инструмент, observation возвращается модели, а
до и после действия сохраняется hash состояния. На Qwen3-14B в ClearML:

- 19 оценённых траекторий, 3 success и 16 failure;
- 301 executed tool actions;
- 4/17/36 действий на траекторию (min/median/max);
- один пример пропущен из-за unsupported oracle tool schema.

HTC обучался только на out-of-fold predictions:

| Метод | AUROC | AUPRC | Risk@80% |
| --- | ---: | ---: | ---: |
| HTC-Full | 0.833 | 0.972 | 0.812 |
| HTC-Reduced | 0.812 | 0.966 | 0.812 |
| No-selection baseline | 0.500 | 0.842 | 0.842 |

HTC показывает promising ranking signal, но practical selective improvement
пока мал: risk при coverage 80% снизился только с 0.842 до 0.812. Из-за всего
трёх успешных траекторий оценка имеет высокую дисперсию и не должна подаваться
как финальный benchmark result.

SAUP и UProp пока не запускались на этом artifact: greedy rollout не содержит
необходимых им Monte Carlo alternatives, situation weights и predecessor
distances. Эти данные не подменялись эвристиками.

## 5. Проверка реализации

- Decoder constraint-mass проверен против brute-force allowed mass: 8/8 тестов.
- JSON grammar, включая optional arguments: 7/7 тестов.
- Формулы и feature extraction SAUP/UProp/HTC: 13/13 тестов.
- Multi-turn pipeline прошёл end-to-end ClearML run с официальным BFCL
  executor и сохранил воспроизводимый rollout artifact.
- Predictions и labels после добавления semantic-token полей для 0.5B
  полностью совпали с предыдущими clean-label artifacts.

## 6. Что уже можно сообщить дипломному руководителю

1. Сформирована актуальная постановка: uncertainty формально валидных действий
   в stateful multi-step tool-use agents.
2. Экспериментально подтверждён constraint tax: hard constraints повышают
   общую валидность и accuracy, но часть invalid outputs превращают в
   исполняемые wrong-valid действия.
3. На реальном BFCL constraint pressure не превзошёл простые log-prob методы.
   Это честный отрицательный результат, воспроизведённый на 0.5B и 3B.
4. Multi-sample AST entropy и INSIDE также не превзошли G-NLL-SMT.
5. Построен настоящий stateful BFCL multi-turn runner и получен первый rollout
   Qwen3-14B с 301 выполненным действием.
6. HTC показывает предварительный trajectory-failure signal, но для устойчивого
   вывода нужна существенно большая и более сбалансированная выборка.

## 7. Следующие обязательные эксперименты

1. Увеличить BFCL multi-turn выборку и выделить независимый test split.
2. Добавить MC action alternatives и distance logging для честных SAUP/UProp.
3. Воспроизвести semantic-token и AST protocol из UQ for LLM Function-Calling.
4. Логировать feasible mass/projection tax и сравнить standard constrained
   decoding с DCCD или близким draft-conditioned baseline.
5. Оценивать не только failure detection, но и Value of Steering: помогло ли
   вмешательство на выбранном шаге и сколько успешных траекторий оно испортило.
6. Добавить второй stateful benchmark, предпочтительно ToolSandbox или
   tau-bench, чтобы вывод не зависел только от BFCL.

Текущая стадия проекта: proof of concept и instrumentation завершены; получен
первый настоящий multi-step результат. До итоговых выводов диплома остаются
масштабирование выборки, SAUP/UProp и эксперимент с intervention policy.
