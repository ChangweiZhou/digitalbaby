# 第二轮 设计 审核 试跑 修订

设计：固定8种可进入DEV/CONFIRM的物理配置，验证冻结、1STEP、ADAPT和FB；用相同前向状态比较完整与截断先前记录信用；固定活动下检验native读出/写入的精确导数。

审核：no-write必须同时覆盖native facts、theta及optimizer；FB只能取pre-write rm且对当前target/punishment不敏感；ADAPT不能等同母体adapt；冻臂不得重新抽差的出生参数。

试跑：attempt1完成72条技术记录，全部通过。跨记录梯度差范数0.0509597；精确读出导数误差3.85e-9，native写入导数相对误差5.42e-7。全部8配置非零状态禁写、只读自主生成、参数计数及反馈幅度/来源通过。

修订：原生基线双读出使用分开的逻辑CPU台账；禁用不需要代理梯度时的重复可微投影；checkpoint包含source identity和完整theta/optimizer/front/time；第三轮补充显式STE有限差分及真实硬KC变化，不把代理梯度当行为进步。
