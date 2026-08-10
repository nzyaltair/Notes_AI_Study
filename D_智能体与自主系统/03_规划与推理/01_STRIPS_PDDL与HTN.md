# STRIPS、PDDL 与 HTN

STRIPS 用动作前置条件、增加列表和删除列表描述状态变化；PDDL 将其标准化；HTN 将高层任务递归分解为子任务和原子动作。

例如 `book_hotel(city)` 可要求 `destination(city)` 与预算可用，并产生已预订及预算更新等效果。LLM 可将自然语言转为候选计划，但需由类型、前置条件、约束求解器或真实工具验证。

References: Fikes & Nilsson, *STRIPS*; Ghallab et al., *Automated Planning*.