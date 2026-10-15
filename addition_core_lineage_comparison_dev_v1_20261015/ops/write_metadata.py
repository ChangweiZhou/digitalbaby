import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import data
from io_utils import write,read,now
import bootstrap

root=bootstrap.HERE
source=read(root/'DESIGN.json')
versions=[]
for item in source['versions']:
    native='151' in item['id']
    versions.append({**item,'implementation':'core.Native' if native else 'core.Recurrent',
        'output_dependency':'source native fast/slow/adapt -> fixed native association readout ->16 values ->common fixed code' if native else 'learned byte embeddings/recurrent actor weights ->learned16-value head ->common fixed code',
        'not_inherited':['No COV-R/SUPPORT role or numeric feature branch','No V9 byte-prediction w/bias used',
                         'No attention/Transformer/ELM','C16 is a new registered adaptation; only C4 source compatibility is asserted'],
        'persistent_state':'16 or32 independent source fast/slow/adapt arrays' if native else 'source-derived embedding/recurrent actor/C16 parameters and RMS moments; no critic/PG'})
write(root/'VERSION_LINEAGE.json',dict(utc=now(),versions=versions,
      scope='Technical five-way addition comparison; no automatic historical capability inheritance or core adoption'))
manifest={}
for world in data.TECH_WORLDS:
    short=world==data.TECH_WORLDS[0]
    d=data.prepare(world,128 if short else 16384,32 if short else 1024)
    manifest[str(world)]=dict(seed=d['seed'],groups=d['groups'],old_records=len(d['old']),new_records=len(d['new']),
                             checkpoints=d['checkpoints'],stream_seeds=dict(old=(world<<8)+1,new=(world<<8)+2,shuffle=(world<<8)+3))
write(root/'DATA_MANIFEST.json',dict(science=False,technical_worlds=manifest,formal_worlds_authorized=False,
      full_actual_streams='retained in each accepted receipt; no HELD teaching'))
(root/'README.md').write_text('''# 加法核心谱系比较：三轮技术练习

实际进度以 results/STATUS.json 为准。EXPERIMENT.md 和 DESIGN.json 是保留的设计合同副本，不能当作运行收据。正式科学实验与 GitHub 上传均未获本轮授权。

五个固定版本：N151-C16、C151-C16、RC151-C16、KEEP-C16-CE、G62-C16-CE。前三者直接执行原生 Full151 方程，固定 FE0/CONTENT 属于历史对照；后两者是明确独立的递归路线。共同 C16 头是新适配，不继承旧输出头的能力结论。

Cycle 1：源码、状态、C4 兼容、CE 更新与禁写检查已通过。Cycle 2：五版 / 15 历史 / 2400 记录已完成并审核；短剂量成绩很弱，详见 cycles/cycle2/REPORT.md。Cycle 3：两个完整技术世界 / 十个作业 / 522240 记录应用，按状态文件准入与执行。

目标仍是从反馈学习加法，以更少训练样本和算力获得未见组合泛化；不加入预置算术规则，不以接口资格或已教题回忆替代泛化。

本地只读监视器：http://127.0.0.1:8921/ 。每轮 source 快照、修订记录与技术收据保存在 cycles/。最终生成 REPORT.md、FINAL_AUDIT 与结果包；不会自动启动科学实验或上传。
''')
