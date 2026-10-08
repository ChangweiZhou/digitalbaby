# 第二轮结果、审核与第三轮修订

DEV822101完整life完成8个训练历史。新版B：旧获取/day2保持/day3新获取/day3修订/day3未改内容均100%；A对应100%/96.875%/100%/87.5%/93.75%。W全训练bpb为A1.781680、B1.598822。仅一个DEV，不作采用或功效证据。

首次独立审计因sorted JSON字典阶段顺序错误失败，实际生命史按正确clock执行。错误保存AUDIT_FAILURE.json；改为独立固定时间顺序后对同一receipt逐行审核，未重跑life。12个receipt negative controls实际被拒绝，包括N_OLD/N_REV真实写入剂量、未来地址、错byte/误差/读出、wrong birth和science namespace。原子待观察checkpoint A/B均逐位等价。

运行前审核进一步发现B clone共享纯地址缓存：数学内容不变，但只读probe会给训练主模型预热缓存，可能影响计时。第三轮改为复制纯地址缓存，与A相同；固定资产仍共享。该修订不改更新式/参数/输入/clock，需再测试cache isolation。

第三轮补齐独立原始fixture重建、错source/byte/pending/计数checkpoint tamper；最终两个新DEV完整life，评估预定行为资格，未通过则不发射。计数矩阵仍是第二轮的固定版本，不再挑η或替代候选。
