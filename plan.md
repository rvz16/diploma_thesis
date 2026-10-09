Слишком высокую уверенность отлавливать попробовать
Добавить метрику критической ошибки модели, когда она уверенна, но дает неправильный ответ прям
Есть бенчмарки, где любой вызов, испралвние тестовго файла является ошибкой
Impossible Bench (Логическое несоответсвие с тестами и решениями модели) Взять оттуда какие-то примеры

Дополнительные бенчмарки для замеров, кроме BFCL:
1). ToolSandbox
2). When2Call
3). tau-bench
4). ComplexFuncBench
5). API-Bank
6). Impossible Bench

## Статус

- [x] Добавлена метрика критической ошибки: wrong-valid вызов, который прошёл
  политику selective execution как низкорисковый.
- [x] Посчитаны critical-error rate, selective risk и доля пропущенных ошибок
  при coverage 20%, 50% и 80% на BFCL multiple/live_simple для Qwen2.5-0.5B/3B.
- [x] Проверена буквальная высокая token-confidence на порогах 0.80/0.90/0.95.
- [x] Перегенерированы BFCL multiple/live_simple для Qwen2.5-0.5B с
  `avg_sem_tok_unc`; predictions и labels полностью совпали со старыми файлами.
- [x] Собран BFCL multi-turn rollout runner на текущем официальном
  `BFCL_v4_multi_turn_base` (V3 multi-turn protocol): constrained action,
  official stateful executor, tool observation, state hash до/после и финальная
  trajectory label. Закреплены data, answers и tool docs; unit/integration
  instrumentation test проходит.
- [x] Добавлен trajectory-level экспорт для HTC: последовательности action-token,
  top-1 и top-5 confidence. SAUP/UProp не запускаются на greedy rollouts до
  добавления требуемых ими MC alternatives / situation weights.
- [ ] Повторить semantic-confidence прогон для Qwen2.5-3B.
- [x] Запущен BFCL multi-turn smoke на Qwen3-14B: 19 валидных траекторий,
  301 executed actions, 3/19 success; сохранён ClearML artifact с
  `action -> observation -> state` и confidence по каждому action.
- [x] На реальном multi-turn artifact посчитан HTC: Full OOF AUROC 0.833,
  AUPRC 0.972; Reduced OOF AUROC 0.812, AUPRC 0.966. Это предварительный
  3-fold smoke result (только 3 successes), не финальная benchmark-оценка.
- [x] Реализован leakage-free Jev (`~typesafe/jev-latest`) pre-execution gate:
  нативный `Choice(execute, review)` получает историю, доступные tools и
  предложенный structured call, но не видит текущий observation или BFCL label.
  Есть deterministic sampling, checkpoint/resume, тесты и парная оценка с
  G-NLL-SMT/CP на одних action rows.
- [ ] Запустить Jev smoke, затем все 301 action и сравнить AUROC/AUPRC,
  risk--coverage, Brier/ECE и latency. Первый официальный API-вызов 2026-10-09
  дошёл до `/v1/systemone`, но OpenRouter вернул `402 Insufficient credits`;
  нужен баланс OpenRouter либо `TYPESAFE_API_KEY` и TypeSafe endpoint.
- [x] Добавлены бесплатные open-weight Jev-compatible альтернативы: полный
  `Cloudflare/clef` 27B (Apache-2.0) и reasoning-модель `PostHog/jeeves` 9B.
  Их pinned checkpoints запускаются локально на ClearML A100 80 GB без
  model-API keys и используют тот же leakage-free `execute/review` protocol.
- [ ] Одинаковый smoke на 5 actions запущен 2026-10-09: первый Clef task
  `142364cfb0474bd2bcf192743545db6c` завершился до model load из-за
  отсутствующего S3 driver в окружении; исправленный retry
  `8fe1d1f9e6524836a8be8cbc571392e7` уже работает на `aiagent01:gpu0`.
  Jeeves task
  `ba11d0485e8d4501983aad9a966249e9` на `aiagent02:gpu0`. После проверки
  артефактов расширить обе модели до общей выборки 40/301 actions.
- [ ] Добавить MC sampling и action-distance logging для честных SAUP/UProp
  trajectory baselines; затем провести full V3/V4 multi-turn evaluation.
- [ ] Добавить abstention/impossible примеры и затем выбрать один внешний
  benchmark из списка выше для полноценного нового запуска.

## Актуальная литература: structured output, UQ и многошаговые агенты

### Прямые работы — обязательны для related work и baselines

1. **SAGE / Structured Uncertainty guided Clarification for LLM Agents**
   ([ACL Findings 2026](https://aclanthology.org/2026.findings-acl.2028/)).
   Моделирует uncertainty по кандидатам tool call и их параметрам, выбирая
   между вызовом инструмента и уточняющим вопросом; вводит multi-turn
   ClarifyBench. Это самый близкий конкурент по structured uncertainty.
   Отличие нашего фокуса: constrained decoding, wrong-valid действие и
   распространение ошибки через состояние среды после выполнения.

2. **Beyond Single-Turn Confidence: Trajectory-Adapted Uncertainty
   Quantification for LLM Agents**
   ([arXiv 2026](https://arxiv.org/abs/2608.11552)).
   Сравнивает action-token probability, trajectory self-consistency и
   self-assessment на BFCL-V4 и $\tau^2$-bench. Нужна как ближайший протокол
   сравнения способов агрегации uncertainty по шагам.

3. **RUPA / From Sequence to Structure: Relational Uncertainty Propagation
   for LLM Agents** ([arXiv 2026](https://arxiv.org/abs/2608.16002)).
   Представляет execution history как граф reasoning states, tool interactions
   и environment feedback, затем propagates risk по зависимостям. Близка к
   H2, но не изучает constrained structured output как причину ошибки.

4. **ToolChain-CRC: Conformal Risk Control for Agentic AI Under Retrieval and
   Tool-Use Drift** ([arXiv 2026](https://arxiv.org/abs/2606.18467)).
   Trajectory-level accept/intervene policy и anytime alarm. Источник для
   строгой постановки selective execution: риск принятой части траекторий,
   а не только AUROC.

5. **HTC / Agentic Confidence Calibration**
   ([arXiv 2026](https://arxiv.org/abs/2601.15778)).
   Уже реализован в `trajectory_baselines.py`; основной feature-based
   trajectory-calibration baseline.

6. **UProp: Investigating the Uncertainty Propagation of LLMs in Multi-Step
   Agentic Decision-Making** ([arXiv 2025](https://arxiv.org/abs/2506.17419)).
   Уже реализован. Разделяет intrinsic uncertainty текущего решения и
   extrinsic uncertainty, унаследованную от прежних действий.

7. **SAUP: Uncertainty Propagation on LLM Agent**
   ([ACL 2025](https://aclanthology.org/2025.acl-long.302/)).
   Уже реализован. Более простой trajectory baseline с situation-aware
   весами для per-step uncertainty.

8. **Uncertainty Quantification in LLM Agents: Foundations, Emerging
   Challenges, and Opportunities**
   ([ACL 2026](https://aclanthology.org/2026.acl-long.738/)).
   Базовая общая постановка Agent UQ: heterogeneous entities, uncertainty
   dynamics и дефицит fine-grained agent benchmarks.

9. **Uncertainty Quantification for LLM Function-Calling**
   ([arXiv 2026](https://arxiv.org/abs/2604.22985)).
   Самая прямая одношаговая работа для нашего action-level baseline: сравнивает
   single- и multi-sample UQ для function calls, AST-based clustering и
   uncertainty только по семантически значимым токенам. Её протокол нужно
   воспроизвести до заявления преимуществ trajectory aggregation.

10. **The Hidden Cost of Structured Generation in LLMs: Draft-Conditioned
    Constrained Decoding** ([arXiv 2026](https://arxiv.org/abs/2603.03305)).
    Формализует projection tax hard-constrained decoding: малая feasible mass
    может направлять модель в локально допустимый, но семантически неверный
    output. Это прямое обоснование нашей цепочки
    `constraint -> wrong-valid action -> state corruption` и baseline DCCD.

11. **From Uncertainty to Action: Learning to Steer LLM Agents**
    ([arXiv 2026](https://arxiv.org/abs/2610.09115)).
    Показывает, что uncertainty может предсказывать неуспех, но не обязательно
    правильный момент вмешательства; вводит Value of Steering и harm-budgeted
    trigger. Нужна как ближайший baseline для selective intervention.

### Полезные дополнительные работы

- **The Confidence Dichotomy: Analyzing and Mitigating Miscalibration in
  Tool-Use Agents** ([ACL 2026](https://aclanthology.org/2026.acl-long.520/)):
  evidence/search tools могут усиливать overconfidence, deterministic
  verification tools — снижать её. При анализе BFCL V4 разделять типы tools.
- **Confidence Estimation for LLMs in Multi-turn Interactions**
  ([ACL Findings 2026](https://aclanthology.org/2026.findings-acl.1280/)):
  per-turn calibration, monotonicity confidence и InfoECE; источник метрик
  для накопления информации, хотя это dialogue, а не tool-use benchmark.
- **Agentic Uncertainty Quantification**
  ([arXiv 2026](https://arxiv.org/abs/2601.15703)):
  uncertainty-aware memory/reflection как active policy, а не только detector.
- **Trajectory UQ taxonomy and TC-ECE**
  ([arXiv 2026](https://arxiv.org/abs/2609.07395)):
  Trajectory-Checkpoint ECE и оценка calibration на разных длинах траектории.
- **Grammar-Aligned Decoding**
  ([arXiv 2024](https://arxiv.org/abs/2405.21047)):
  показывает, что обычное grammar masking искажает условное распределение;
  теоретическая опора для анализа constrained probabilities.
- **Clarify the User or Verify the World?**
  ([arXiv 2026](https://arxiv.org/abs/2609.32255)):
  uncertainty routing между ACT, CLARIFY и VERIFY на tool-use trajectories.
- **JSONSchemaBench**
  ([arXiv 2025](https://arxiv.org/abs/2501.10868)):
  benchmark покрытия JSON Schema, скорости и качества constrained generation.
- **CONSTRUCT / Real-Time Trustworthiness Scoring for LLM Structured Outputs**
  ([arXiv 2026](https://arxiv.org/abs/2603.18014)):
  black-box output- и field-level uncertainty без logprobs; полезный baseline
  для structured-output detection вне open-weight setting.
- **Knowing What You Know Is Not Enough**
  ([arXiv 2025](https://arxiv.org/abs/2511.13240)):
  демонстрирует action--belief gap: статическая confidence не гарантирует
  согласованное действие агента.
- **Uncertainty of Thoughts**
  ([arXiv 2024](https://arxiv.org/abs/2402.03271)) и **Conformal Information
  Pursuit** ([arXiv 2025](https://arxiv.org/abs/2507.03279)):
  uncertainty-aware и conformal information-seeking baselines для
  последовательного уточнения информации.
- **BFCL V3/V4 documentation**
  ([official repository](https://github.com/EnlightenedAI/BFCL/blob/main/berkeley-function-call-leaderboard/bfcl_eval/data/README.md)):
  V3 даёт Multi-Turn/Multi-Step; V4 Agentic — persistent state, Web Search и
  Memory. Это следующий benchmark после текущих одношаговых BFCL artifacts.

### Обновлённое позиционирование диплома

Не заявлять «первая UQ-работа для многошаговых агентов». Более точная ниша:

> Hard structured constraints can transform a retryable invalid action into a
> schema-valid but semantically wrong state transition. The project measures
> whether this wrong-valid action can be detected before execution, how its
> risk propagates through a multi-step tool-use trajectory, and when the
> agent should intervene.

То есть новизна — не ещё один общий uncertainty score, а связка:
`constrained structured decoding -> wrong-valid action -> state corruption ->
critical trajectory error -> selective intervention`.

### Следующие действия по литературе и экспериментам

- [ ] Детально прочитать SAGE и зафиксировать точное отличие по данным,
  uncertainty signal и intervention policy.
- [ ] Воспроизвести action-level protocol из UQ for LLM Function-Calling:
  semantic-token score и AST clustering как одношаговые baselines.
- [ ] Измерить feasible probability mass / projection tax hard constraints и
  сравнить обычный constrained decoding с draft-conditioned вариантом DCCD.
- [ ] Разобрать Beyond Single-Turn Confidence: их action spans, aggregators,
  datasets и сильные baseline results; воспроизвести совместимые metrics.
- [ ] Сверить RUPA и ToolChain-CRC с H2--H4, чтобы не повторить их постановку
  propagation/intervention без собственного constrained-output вклада.
- [ ] Сопоставить selective execution с Value of Steering: отдельно измерять
  detection failure и полезность вмешательства при заданном harm budget.
- [x] Собрать BFCL V3 Multi-Turn runner с action/state/observation logging.
- [x] После GPU rollout запустить HTC как реальный baseline.
- [ ] Добавить MC alternatives/situation weights и затем честно запустить
  SAUP/UProp.
- [ ] Добавить TC-ECE/InfoECE как secondary calibration metrics alongside
  critical-error rate and risk--coverage.
