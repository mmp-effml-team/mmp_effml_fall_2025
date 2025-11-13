## `Материалы кафедры ММП факультета ВМК МГУ. Введение в эффективные системы глубокого обучения.`

## `[БОНУСНОЕ] Задание 05. Context Parallel Attention`

Дата выдачи: <span style="color:red">__13.11.2025 12:00__</span>.

Мягкий дедлайн: <span style="color:red">__28.11.2025 23:59__</span>.

Стоимость: __5 баллов__.

#### `Москва, 2025`

Контекстный параллелизм можно считать разновидностью методов Data Parallel, с той лишь разницей, что в блоке attention нужно как-то объединить разнесённые по разным GPU фрагменты одной последовательности. Подробнее о контекстном параллелизме можно почитать здесь:
https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=context_parallelism

В небольшом БОНУСНОМ домашнем задании вам предстоит реализовать два метода:

1. AllGather context parallel (2 балла)
- Собираем KV и перемножаем с локальным Q

2.RingAttention (3 балла)
- С помощью P2P-операций (send/recv) пересылаем блоки KV ([paper](https://arxiv.org/pdf/2310.01889))
- Используем онлайн-softmax для финальной агрегации ([paper](https://arxiv.org/pdf/1805.02867))

Задание мы выдаём впервые; обязательно пишите (tg: `@serv01`), если у вас будут вопросы по выполнению.

Запуск:

```
cd task5
uv venv
source .venv/bin/activate
uv pip install torch numpy
python3 main_todo.py
```