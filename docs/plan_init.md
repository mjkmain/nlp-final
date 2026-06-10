# GCT799 (NLP) final project 

## Task description

Problem: LLM은 재미있는 농담, humor를 하지 못한다.

Why Is Humor So Hard?
- Humor is highly culture-specific — and often language-specific.
- It also demands creativity.
- LLMs are notoriously weak at both, which makes humor an excellent testbed for benchmarking their capabilities.

Task: How can we make LLMs funny? 

We are going to focus on Korean humor, oogiri task.

## My opinion

"왜 LLM이 재미있는 humor를 하지 못할까?" 라는 것은 "사람이 어떻게 농담을 하는지, 왜 웃는지"에 집중해봐야 한다. 
어떤 상황에서 일반적으로 예측할 수 있는 말을 하게 되면 웃기지 않는다. 웃긴 말을 하기 위해서는 예측할 수 없는, 창의적인 말을 해야 한다. 
하지만, LLM은 수십T 토큰으로 학습되며 가장 일반적인 다음 토큰을 예측하게 된다. 이 것이 LLM이 웃길 수 없는 이유라고 생각한다. 

이러한 관점으로 token decoding 관점에서 접근해보면 어떨까? 

기본적인 틀을 GRPO/BoN sample 등의 preference learning 활용으로 잡아보자.
만약 LLM이 특정 시점의 토큰 generation에서, top-k token이 아닌 N-th ~ (N+M)-th 사이의 token을 선택하면 과연 창의적인 LLM이 될 수 있을까? 이를 creative token decoding이라고 하자.
물론 이 방법은 문장 붕괴 등의 우려가 있기 때문에 적절한 threshold가 필요하며, 적절한 위치에서의 작업이 필요하다. 또한 preference learning의 기본 틀인 roll-out이 필수적이다. 

정리해보면, 현재 생각하고 있는 프레임워크는 
- 유머생성 시, LLM의 creativity 향상을 위해 creative token decoding을 적용한다. 
- 생성된 유머들은 roll-out으로 활용되어 preference learning을 수행한다. preference learning에서는 외부 모델을 사용하여 유머 평가를 진행한다. 
- 좋은 유머를 생성하기 위해서는 HP가 높은 seed가 필수적이다. HP를 측정하여 web source를 모으는 작업을 수행한다.


## Detailed Plan 

- model: `google/gemma-4-E2B-it` (VLM)
- Dataset: zhongshsh/CLoT-Oogiri-GO (한국어 번역 필요, Qwen3.5-122B-A10B)
- SFT 진행 (cold start) 
- SFT된 모델에 5개의 샘플의 4개 roll out을 뽑아서 팀원에게 전달 -> 6명의 팀원이 유머 점수 라벨링 
- reward model의 in-context로 이 유머 점수를 활용, persona-aware reward 기대
    - Problem: image input in in-context
