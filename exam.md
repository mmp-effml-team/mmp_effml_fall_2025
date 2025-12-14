## `Вопросы к экзамену`

1. Основы архитектуры GPU. Модель программирования CUDA. Алгоритм параллельной редукции
1. Mixed Precision Training
1. Kernel Fusing. Flash Attention
1. Постановка задачи инференса. Спекулятивный декодинг. EAGLE
1. Постановка задачи инференса. Квантизация. GPT-Q & SmoothQuant
1. Постановка задачи инференса. Квантизация. Quantization Aware Training
1. Постановка задачи инференса. Pruning. Structured Pruning of LLMs
1. Постановка задачи инференса. Дистилляция. Knowledge Distillation & MiniLLM
1. Постановка задачи инференса. Дистилляция. Knowledge Distillation & DINO
1. Efficient Finetuning. Aдаптеры. LoRA & QLoRA. Gradient Accumulation & Gradient Checkpointing
1. GPU Clusters. Соединение серверов с GPU. Топологии соединения. Hardware варианты пересылок GPU-to-GPU. Метрики стабильности обучения
1. DDP. Связь с Gradient Accumulation. Allreduce и его виды (RingAR, TreeAR, DoubleTreeAR). Как перекрыть оптимизацию с backward?
1. FSDP. Перекрытие, prefetch. Разница FSDP1.0 vs FSDP2.0
1. FSDP. DeviceMesh и DTensor, redistribute.
1. Parallelisms. Tensor Parallel, Sequence Parallel 
1. Parallelisms. Идея MoE, Grouped Gemm, Expert Parallel. Разница с DP и TP
1. Context Parallel, Как считать attention блоками? Ring Attention и Deepseed Ulysses

## `Теоретический минимум`
1. Архитектура трансформера
1. Основные причины замеделения обучения (compute bound, CPU CUDA API Bound, Synchronization Bound) и как с ними бороться
1. Основная идея спекулятивного декодинга
1. Основная идея Optimal Brain Quantization
1. Straight Through Estimator
1. Как добиться от pruning-а ускорения на железе?
1. Основная идея Knowledge Distillation
1. Почему в MiniLLM используется Reversed KL?
1. LoRA
1. Gradient Accumulation как работает и в какои случае нужен?
1. Gradient Checkpointing как работает и в какои случае нужен?
1. В чем отличие intra / inter node communications?
1. Что такое backend и frontend networking? В чем разница?
1. Коллективные операции в NCCL
1. ZeRO идея. Разница между Stages
1. Сколько памяти нужно для обучения модели размера M без учета активаций?
1. Идея MoE
1. В чем разница между DP / TP / EP / CP
