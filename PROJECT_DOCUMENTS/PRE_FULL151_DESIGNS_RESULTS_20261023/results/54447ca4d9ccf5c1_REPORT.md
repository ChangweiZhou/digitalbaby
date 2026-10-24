# 固定地址、2/4/8存储分区：完整实验结果

224个独立世界，896条正式轨迹；常规96，高负载128。以下统计只在全部配对世界完整后计算。

|模型|主要收益|普通保护|常规保留风险|高负载保留风险|候选|
|---|---|---|---|---|---|
|FB2|PRIMARY_GAIN_CONFIRMED|NO_CONFIRMED_DENSE_GUARD_HARM|STABLE_RISK_UNRESOLVED|STABLE_RISK_UNRESOLVED|ADVANCE_AS_RESOURCE_CANDIDATE|
|FB4|PRIMARY_GAIN_CONFIRMED|NO_CONFIRMED_DENSE_GUARD_HARM|STABLE_RISK_UNRESOLVED|STABLE_RISK_UNRESOLVED|ADVANCE_AS_RESOURCE_CANDIDATE|
|FB8|PRIMARY_GAIN_CONFIRMED|NO_CONFIRMED_DENSE_GUARD_HARM|STABLE_RISK_UNRESOLVED|STABLE_RISK_UNRESOLVED|ADVANCE_AS_RESOURCE_CANDIDATE|

|高负载主要比较|差值pp|同时区间pp|自由度|
|---|---:|---|---:|
|FB2 − B|7.585|[5.623, 9.547]|60.622727040163724|
|FB4 − B|17.025|[14.926, 19.124]|54.30437138213237|
|FB8 − B|25.879|[23.551, 28.207]|56.25575125111531|
|FB4 − FB2|9.440|[7.418, 11.462]|56.169210359216265|
|FB8 − FB4|8.854|[7.182, 10.526]|55.4694605278561|

最小推进的已测试分区数：2。未确认收益不意味着等价；推进标签不证明安全，也不自动替换父模型。

![资源曲线](RESOURCE_CURVE.png)

主要5项/普通保护45项/稀疏风险6项，alpha分别.02/.02/.01。高负载32个字典×轮换固定层，每层4世界；常规4层，每层24世界。t/Satterthwaite为近似推断；完整层均值、SE和自由度已导出。

保留风险使用预定Hoeffding区间，常规/高负载半宽38.43/33.28个百分点。未决不是安全；无共同已知支持不是成功保留。

内容地址固定；选中的存储保留完整原生B反馈与分配，其他分区按同一物理时间被动衰减。额外存储容量与隔离同时改变，本轮不能识别两者各自因果作用。

配对和事件等权碰撞的单调性来自嵌套结构。质量加权碰撞按支持查询等权，合并质量比例另列；年龄是cue暴露年龄，重复查询不是额外独立样本。

所有原始世界、紧凑世界回执保留在本地；分享包仅包含紧凑结果、实现、冻结合同与重建清单，不复制输入世界、历史资产或缓存。

主轨迹CPU进程墙钟秒数合计17093.6，不是四worker调度总时长。
