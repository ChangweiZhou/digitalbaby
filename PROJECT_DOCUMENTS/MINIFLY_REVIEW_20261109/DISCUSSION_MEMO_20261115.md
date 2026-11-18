# Discussion memo, 14 to 15 November 2026

From the outside reviewer, at Mr. Xu's request ("Push all the discussions as a long memo in the repo"). It continues `AUDIT_NOTES_MEMO_20261114.md`, which ends at 05:26 UTC on 14 November.

This is a record, not a summary: every message of Mr. Xu's and every substantive reply of mine since then, verbatim from the session transcript, in order, with UTC times. Short progress lines, tool output and automatic notifications are left out. Headings inside a reply are set three levels lower. Paths to uploaded files are replaced by a description. Later replies correct earlier ones in several places; the corrections are in the replies themselves and are not applied backwards.

Where the discussion ended, in five lines:

- The goal is a continual learner: one stream read once, fixed memory and computation, no full backpropagation, judged late in a long life so that a slow start followed by overtaking can be seen.
- Sweep, Kolen–Pollack feedback and backpropagation build the same answer code and lose the same thing in a life (the unrehearsed family's modulus); the rule has not been the difference.
- Three full-scale jobs are briefed and audited (E36, E37, E38); E34's confidence gain passed on one seed and learns slowly.
- The programme from here is Part II of `MEMO_TO_COORDINATOR_LEARNED_STRUCTURE_20261115.md` (revision 4): a small world, long lives, trunks and branches, probes, and a first job that asks whether the conventional learner gets stuck.
- Documents of the period, all in this folder: that memo, `AUDIT_E36_E37_E38_BRIEFS_20261115.md`, and the check scripts named in them.

## Contents

1. 11-14 14:26 — Mr. Xu: Take a look at current state, discuss next steps.
2. 11-14 14:32 — Reviewer: **Nothing has run for about nine hours.** Both E31 and E32 are waiting
3. 11-14 16:34 — Mr. Xu: Deliver the message to worker A and B as I am outside.
4. 11-14 16:37 — Reviewer: Both messages are delivered and both sessions woke within a minute.
5. 11-14 16:37 — Mr. Xu: Thank you
6. 11-14 17:40 — Reviewer: Both workers picked up within a minute, both checks passed, and E31 st
7. 11-14 18:44 — Mr. Xu: Is everything on track?
8. 11-14 18:45 — Reviewer: Yes. As of 18:44 UTC nothing has failed or stalled, and the timing mat
9. 11-14 21:33 — Reviewer: Everything is still on track at 21:33 UTC: 23 of 39 runs are done, non
10. 11-14 22:16 — Mr. Xu: Are the two workers done?
11. 11-14 22:16 — Reviewer: Not yet. As of 22:16 UTC nothing new has been pushed since my last rep
12. 11-15 00:19 — Mr. Xu: Are the two workers done?
13. 11-15 00:19 — Reviewer: Worker A is done; worker B is not. As of 00:19 UTC:
14. 11-15 01:16 — Reviewer: E32 is complete and worker B's report is in; my reader agrees with it 
15. 11-15 02:38 — Mr. Xu: Discuss what we should do next. Should we wait?
16. 11-15 02:41 — Reviewer: **Don't wait.** The one thing still running is worker B's last six E31
17. 11-15 02:49 — Mr. Xu: I am not sure whether the replicate is worth the CPU powers. What is y
18. 11-15 02:50 — Reviewer: My reasoning was weaker than I made it sound, and I think your doubt i
19. 11-15 02:51 — Mr. Xu: The writing above is poor. Use standard academic language, put a bit m
20. 11-15 02:53 — Reviewer: I revised my recommendation on replication between two consecutive ans
21. 11-15 03:06 — Mr. Xu: Message for the reviewer: "Please audit `OPUS_TASK_E33_WHICH_TARGETS_B
22. 11-15 03:12 — Reviewer: # Audit of the E33 brief (commit `00fad472`): three text changes requi
23. 11-15 16:05 — Mr. Xu: Discuss next steps.
24. 11-15 16:10 — Reviewer: ## Summary
25. 11-15 16:24 — Mr. Xu: Both deliveries are merged and E33's endpoints recompute from its read
26. 11-15 16:29 — Reviewer: # Audit of the E34 brief (commit `9291d23b`): sound; one required chan
27. 11-15 16:55 — Mr. Xu: 现在项目在什么状态呢？
28. 11-15 16:56 — Reviewer: 截至 11 月 15 日 16:55（UTC），项目处于"原因已基本定位、第一个补救规则刚获准并开始实验"的阶段。
29. 11-15 16:57 — Mr. Xu: 我不是非常理解，我们反复和backpropagation进行比较，固然能够帮助发现问题，但是似乎无法找到超越backpropagation的
30. 11-15 16:58 — Reviewer: 您的判断是对的：按目前的实验设计，这条路线最多只能在局部性约束下追平反向传播，不可能超越它。
31. 11-15 17:00 — Mr. Xu: 如果你需要更多信息才能做判断，给deep research agent提供一个prompt。如果不需要，进一步讨论这个问题，因为我们已经做了
32. 11-15 17:01 — Mr. Xu: 另外，考虑到将问题变难极为容易，第四个困难显然不是本质的。
33. 11-15 17:02 — Mr. Xu: 此前实验发现了果蝇的结构似乎效果不好，复盘一下此前的决策过程。
34. 11-15 17:05 — Reviewer: 先回应两点。规模那一条我撤回：把任务加难很容易，它不是本质限制。另外，这次复盘不需要外部检索，项目记录本身已经足够；我重读了 11 月 6 
35. 11-15 17:11 — Mr. Xu: 1） 但你没有回答超越backpropagation local learning mechanism是否需要deep research a
36. 11-15 17:23 — Reviewer: 三点依次回答。在检索和实验结果回来之前，我不对"局部机制能否超越反向传播"下结论。
37. 11-15 18:06 — Mr. Xu: I have read the old ring module and the current birth harness. Now che
38. 11-15 18:12 — Reviewer: # 两份 E35 任务书的审查（提交 `907c9fc4`）：设计可行，有四处必须修改
39. 11-15 18:27 — Mr. Xu: (an uploaded literature report) Discuss next steps given the research 
40. 11-15 18:36 — Reviewer: **Position.** Under a fixed memory and compute budget, a local rule wi
41. 11-15 18:40 — Mr. Xu: I repeatedly said as long as we do not do full backpropagation, anythi
42. 11-15 18:41 — Reviewer: You are right, and I will stop asking. The rule is settled: anything s
43. 11-15 18:54 — Mr. Xu: (an uploaded literature report)
44. 11-15 18:57 — Reviewer: The second report closes the literature question: no further research 
45. 11-15 18:58 — Mr. Xu: So now given the circuit structure itself is shaped by learning in our
46. 11-15 19:25 — Reviewer: **Your premise holds, and the saved networks show it.** I read the six
47. 11-15 19:42 — Mr. Xu: Write an appropriate memo for the coordinator to implement.
48. 11-15 19:51 — Reviewer: The memo is written and pushed to `claude/vibrant-cerf-sje726` at `4fd
49. 11-15 19:53 — Mr. Xu: We want to avoid back and forth, E33 and E34 has shipped. See if this 
50. 11-15 19:58 — Reviewer: E33 and E34 change nothing in jobs A, B and C. They add two small item
51. 11-15 20:01 — Mr. Xu: 现在项目处于什么状况？
52. 11-15 20:01 — Reviewer: 项目现在处在一个换方向的节点上：两条旧线各出了一个结果，新方向的备忘录已经写好，但协调者还没有动作，默认分支从 15 日 13:16 起没有
53. 11-15 20:08 — Mr. Xu: 从哲学角度上来说，最朴素的反向传播只对每个样本进行同样的学习，通过误差进行逐级修正，网络结构是恒定的，虽然可能在RL或其他阶段只是激发网络的
54. 11-15 20:12 — Reviewer: 您的表述在文献里有直接对应，检索没找到，主要是因为我把检索问题问错了层次。实验慢，是因为两周来多数实验在回答另一个问题，而且元层次的搜索一直
55. 11-15 20:12 — Mr. Xu: Send an agent to work on this: "Reviewer: “Please audit the three brie
56. 11-15 20:13 — Reviewer: An agent is now auditing the three briefs (E36, E37, E38) at `4dd0b9f7
57. 11-15 20:16 — Mr. Xu: 你可以自己决定实验的快慢，不用等我决定。因为此前目标出现错位，我们先讨论下一步如何做。我们坚持做持续学习。
58. 11-15 20:19 — Reviewer: 下一步我建议先把目标换成一个可以直接计分的量，再围绕它重排实验。之前目标错位，是因为我们量的是"固定训练量之后，在固定题集上的成绩"，比的是
59. 11-15 20:23 — Mr. Xu: 但是这个还是不能回答问题，因为显然可能出现我们的学习者一开始比传统的学习者更慢，但是在持续学习过程中最后做到“后来居上”的问题。只要我们的学
60. 11-15 20:25 — Reviewer: 您说得对，这是我上一个方案的缺陷，要改的是量什么和生命多长。
61. 11-15 20:37 — Reviewer: 审查完成：三份任务书的事实都对，但都不能原样发给工作者，一共要改六处。六处都写成了可以直接照改的文字，改完不需要再审。
62. 11-15 20:42 — Mr. Xu: 现在元问题已经非常清楚，如何在有限的算力和时间限制下设计实验环境和目标，进行有效的持续学习实验迭代，平衡短周期能够发现更多样化信息和长周期得
63. 11-15 20:45 — Reviewer: 我的看法是：不要把短实验当成长实验的缩小版。短实验用来测量学习者的几条"规律"，长实验用来检验这些规律合起来能否预测长期行为。这样两者是分工

## 1. 2026-11-14 14:26 UTC — Mr. Xu

> Take a look at current state, discuss next steps.

## 2. 2026-11-14 14:32 UTC — Reviewer

**Nothing has run for about nine hours.** Both E31 and E32 are waiting on worker A, whose session has not moved since 02:18 UTC, before either set of instructions was written.

##### Where things stand (14:30 UTC, 14 November)

- **E32 brief.** Revision 4 went out at 05:19 UTC with my six changes made as written.
- **E32 main worker (worker B).** By 05:47 UTC it had finished everything it can do alone: the runner, checks 3 to 6, the nine push strengths registered, and the dry births. It has been blocked since 07:48 UTC, asking for the second worker's check records.
- **What I verified of its work just now:**
  - Its zero-strength birth equals E28's floor at rows 4,096 and 8,192 in all 13 tensors, by my own comparison.
  - My numpy reader, written before its runner existed, reproduces all nine registered strengths to about 5 parts in 10 million. The brief asks for 1 in 100,000.
- **From its own records, not rechecked by me:**
  - The gate comparison I worried about came out identical.
  - Its forecast is 19.8 CPU-hours, 6.6 hours of wall time.
- **E31.** Worker B ran stage 1b at 04:17 UTC; worker A never did, so no full life has run. In B's dry lives the three push strengths bracket backprop's range of 43–53% undecided gates:

  | Strength | Undecided share after 1,536 updates (from 65%) |
  |---|---|
  | 0.25 | 21% |
  | 0.05 | 43% |
  | 0.01 | 62% |

- **The misrouted message.** The instruction you sent this session at 03:51 UTC ("do stage 1b, and if the two of you agree…") was most likely worker A's copy. Worker B acted on its copy 26 minutes later.
- **E32's second-worker message** has no visible recipient: no branch or session shows that work started.

##### Next steps

1. **Send worker A both messages now:** E31 stage 1b under addendum 2, and the E32 second-worker checks (check 7). Its E32 part is three dry births, about 15 minutes of compute once the code is written.
2. **Release 12 of E32's 21 births without waiting.** This needs your word or the coordinator's, because the brief says no full birth starts before the second worker agrees.
   - The three ceiling reruns use E28's runner unchanged and are hash-checked at every row.
   - The nine push births have their strengths cross-checked by my reader and their code path shown bit-equal at zero strength.
   - Together they take about 3.8 hours of wall time. G, H and K (about 2.8 hours) wait for worker A's dry births.
3. **Tell worker B not to run Ks at full length.** Your instruction and addendum 3 still conflict.
4. **Order the queue.** Worker B's machine will be full with E32 for six to seven hours. I would run E32 first, since it has the largest effect and the cheapest runs. E31 stage 2 can start on worker A's machine once stage 1b agrees, with B's three E31 arms after E32.
5. **Write `e32_read.py` while the births run.** When the first rows for seed 73129 land, I can check its keeper counts against my reader within the hour.
6. **Do not tie the 17 November close to E32.** The close can already say: level at birth, 37–54% against 63–89% on composition, the gap is in the life update, and without an answer mask the sweep loses its keeping units between updates 4,096 and 12,288 (the last from my unregistered reader). E32 adds which remedy helps, if any.

Still open and cheap: the curvature measurement the literature report suggested, and the repository items in the memo.

I can deliver both instructions to worker A's session myself if you want; say so and tell me whether to include the 12-birth release for worker B.

## 3. 2026-11-14 16:34 UTC — Mr. Xu

> Deliver the message to worker A and B as I am outside.

## 4. 2026-11-14 16:37 UTC — Reviewer

Both messages are delivered and both sessions woke within a minute.

**To worker A** (idle since 02:18 UTC):
- Your 03:51 UTC instruction, word for word: stage 1b under addendum 2, then stage 2 if the two agree, and Ks as a dry run only.
- Its E32 job as second worker: write G, H, K and the strength computation independently, run three dry births, and compute the nine strengths without training.
- Where worker B's records are, and a request to tell worker B when each part is pushed.
- My suggested order, labelled as mine: stage 1b first, then the E32 check while its lives run.

**To worker B** (blocked since 07:48 UTC):
- The answer it was waiting for: worker A had received nothing, and now has both jobs.
- The 12-birth release: start the three ceiling reruns and the nine push births now; hold G, H and K until worker A's dry births agree.
- Your Ks sentence stands until you or the coordinator say otherwise.

Things you should know:

- **The release rests on my reading of your reply.** I had asked whether to include it, and you answered "Deliver the message to worker A and B". I quoted that exchange to worker B in full and told it that holding is a fair call if it judges this is not enough against the brief. If you did not mean to release those births, say so and I will tell it to stop.
- **Worker B runs in default permission mode**, not auto, so it may stop at approval prompts while you are away. Worker A is in auto mode.
- **I scheduled one check for 17:38 UTC.** I will look at both branches and sessions and report to you in a few lines. I will send the workers nothing further unless one is stuck on a plain fact from the repository, such as worker A's commit, which worker B asked to be told about.

## 5. 2026-11-14 16:37 UTC — Mr. Xu

> Thank you

## 6. 2026-11-14 17:40 UTC — Reviewer

Both workers picked up within a minute, both checks passed, and E31 stage 2 and E32's full births are now running or queued. Nothing needs you.

- **Worker A** finished both jobs in 27 minutes (pushed by 17:04 UTC).
- **E31 stage 1b agrees.**
  - The Ks hashes are equal.
  - The three bounded-push arms differ in hash but agree under addendum 2's rule: gradients within 1e-6.
- **E31 stage 2 is running on both machines.**
  - Worker A has G, H and the 0.05 push.
  - Worker B has its three K lives, started at 17:04 UTC.
- **E32 check 7 passed at 17:05 UTC.**
  - Worker A's G, H and K dry births have the same hashes as worker B's.
  - Its nine strengths agree to 5.6 parts in 10 million.
- **All 21 E32 births are queued on worker B.**
  - They start as the K lives free their slots, at about 18:30 UTC, three at a time.
  - The ceiling reruns go first.
  - Worker B paused its other six E31 lives until E32 is done.
- **The 12-birth release was never used.** Worker B held it, and the brief's own condition was met 28 minutes later. So nothing ran on my reading of your reply.
- **No Ks life runs at full length.** Worker B confirmed it.

Rough finish times, from the workers' own estimates:

| Work | Done by (UTC) |
|---|---|
| Worker A's nine E31 lives | about 21:00 today |
| E32's 21 births | about 01:00 on 15 November |
| Worker B's remaining six E31 lives | about 04:00 on 15 November |

I scheduled one more check for 21:30 UTC. By then the ceiling reruns and the first seed-73129 births should be in, and I will check them against my own reader and report.

## 7. 2026-11-14 18:44 UTC — Mr. Xu

> Is everything on track?

## 8. 2026-11-14 18:45 UTC — Reviewer

Yes. As of 18:44 UTC nothing has failed or stalled, and the timing matches the forecasts.

- **E31 lives:** 5 of 18 are complete.
  - Worker B's three K lives finished at 18:15, about 1.2 hours each.
  - Worker A's G and H lives for seed 73129 finished at 18:17 and 18:22. Its third life of that batch has not reported yet.
- **E32 births:** the three ceiling reruns started at about 18:16, a little ahead of the 18:30 estimate. Each has written its first row, and all three match E28's recorded hashes (I compared them from the files).
- **Sessions:** both workers are connected and running their queues in the background.
- **Default branch:** unchanged, so the coordinator has not read any of this yet.

I have only checked that runs complete and hashes match; I have not read any scores. The 21:30 UTC check will do that against my own reader for the first seed-73129 births and report.

## 9. 2026-11-14 21:33 UTC — Reviewer

Everything is still on track at 21:33 UTC: 23 of 39 runs are done, none failed, and neither machine has restarted.

##### Progress

- **Worker A:** 8 of its 9 E31 lives are in. The last one should finish around 22:50 UTC.
- **Worker B:** the three ceiling reruns and all nine G, H and K births are in. The nine push births are next, done around 00:40 UTC; its six remaining E31 lives follow.
- **Ceiling reruns:** all six rows match E28's recorded hashes in all three seeds.
- **My reader:** it reproduces the receipts' own 12- and 20-letter counts exactly for G, H and K (seed 73129), so the numbers below read the files the way the workers do.

The workers judge the predictions; these are first numbers only.

##### E31, lives (end of life, seeds 73129 / 93683 / 116089)

| Arm | New instances of taught pairs | Untaught plus-minus pairs |
|---|---|---|
| Floor (sweep, on file) | 83.5 / 83.8 / 81.0 | 54.2 / 44.5 / 37.2 |
| G, exact gate rows (bound) | 84.7 / 84.7 / 87.0 | 56.7 / 55.6 / 65.6 |
| H, exact recurrent map (bound) | 86.7 / 91.7 / 92.2 | 72.9 / 74.5 / 71.4 |
| K, learned feedback (candidate) | 88.7 / 92.7 / 85.8 | 74.4 / 78.2 / 63.9 |
| Push at 0.05 | 82.5 / 91.8 / — | 50.0 / 73.6 / — |
| Ceiling (backprop, on file) | 93.2 / 97.3 / 93.0 | 72.3 / 82.8 / 73.9 |

- **H and K are 19 to 34 points above the floor on untaught pairs in every seed.** H is within 9 points of the ceiling, K within 10.
- **Single end values are noisy.** Adjacent stages of one life swing by about 10 points.

##### E32, every-byte births (my reader)

Seed 73129, mean of the last three rows:

| Arm | 12 letters | 20 letters | 40 letters | Keepers that write, at 12 |
|---|---|---|---|---|
| Floor | 59.6 | 8.1 | 1.0 | 1 |
| G | 65.1 | 25.4 | 8.3 | 2 |
| H | 81.1 | 55.4 | 4.3 | 9 |
| K | 77.9 | 44.0 | 12.3 | 9 |
| Ceiling | 92.8 | 91.8 | 82.0 | 64 |

- **At 40 letters nothing moves.** G, H and K are at 4–12% here, and 1–12% at the final row of the other two seeds; "moves" needs about 31%.
- **At 20 letters H and K rise in two seeds and not in the third.** Final rows for the other seeds: H 18 and 65, K 11 and 55.
- **None of the three keeps its keepers.** Each starts with 54–55 at update 4,096 and ends with 2–11, while the ceiling goes from 53 to 67. H and K lose them more slowly than the floor.

For seeds 93683 and 116089 only final rows are committed, so I cannot compute the brief's endpoint there; worker B's reader will.

The push births are the arms that act on keepers directly. I scheduled the next check for 01:15 UTC, when they should be in.

## 10. 2026-11-14 22:16 UTC — Mr. Xu

> Are the two workers done?

## 11. 2026-11-14 22:16 UTC — Reviewer

Not yet. As of 22:16 UTC nothing new has been pushed since my last report, which is expected at this point in both queues.

- **Worker A:** 8 of 9 lives done. The last one (the 0.05 push, seed 116089) should finish around 22:50 UTC, after which it writes its report.
- **Worker B:** 12 of 21 E32 births done. The nine push births are running three at a time and should finish around 00:40 UTC. Its six remaining E31 lives start after that and should finish around 04:00 UTC.

My next check is at 01:15 UTC.

## 12. 2026-11-15 00:19 UTC — Mr. Xu

> Are the two workers done?

## 13. 2026-11-15 00:19 UTC — Reviewer

Worker A is done; worker B is not. As of 00:19 UTC:

- **Worker A:** all nine E31 lives are complete and its end-of-life table is pushed (22:49 UTC). It reports P31.1 misses, P31.2 holds and P31.7 misses; I have not checked those judgments yet.
- **Worker B:** 18 of 21 E32 births are in. The last three (the push births for seed 116089) should finish within about 20 minutes. Then come its reader and report, and its six remaining E31 lives, which should take until about 04:00 UTC.

The default branch has not moved, so the coordinator has read none of it. My 01:15 UTC check will read the push births with my own reader and report.

## 14. 2026-11-15 01:16 UTC — Reviewer

E32 is complete and worker B's report is in; my reader agrees with it on every seed-73129 figure I can compute. E31 has six lives left, on worker B (the 0.25 and 0.01 push), due around 04:00 UTC.

##### Checks

- **All 21 births completed** with finite losses.
- **Every push birth** matched E28's row 4,096 hash and recomputed its strength equal to the registered value (I checked all nine receipts).
- **Worker B's reader** reproduced the E30 rows exactly and the floor table within one unit.
- **My reader against its report:** identical on accuracy and keeper counts for G, H, K and the three push strengths on seed 73129, the only seed with rows committed.

##### E32 result (worker B's table, mean of the last three rows)

Accuracy at 12 / 20 / 40 letters, by seed:

| Arm | 73129 | 93683 | 116089 |
|---|---|---|---|
| Floor | 59.6 / 8.1 / 1.0 | 67.2 / 16.6 / 1.6 | 74.0 / 34.1 / 1.3 |
| G | 65.1 / 25.4 / 8.3 | 73.1 / 19.7 / 2.4 | 63.2 / 34.2 / 4.4 |
| H | 81.1 / 55.4 / 4.3 | 75.6 / 28.1 / 2.4 | 80.1 / 57.7 / 6.3 |
| K | 77.9 / 44.0 / 12.3 | 79.2 / 28.6 / 0.9 | 76.9 / 50.3 / 6.4 |
| Push 0.25 | 66.2 / 41.6 / 10.3 | 75.7 / 25.9 / 8.2 | 74.9 / 40.6 / 6.8 |
| Push 0.05 | 62.3 / 16.2 / 1.7 | 69.1 / 12.8 / 1.7 | 78.1 / 31.0 / 0.4 |
| Push 0.01 | 69.2 / 4.6 / 0.6 | 73.2 / 17.4 / 2.7 | 74.7 / 20.4 / 1.9 |
| Ceiling | 92.8 / 91.8 / 82.0 | 95.4 / 93.6 / 82.8 | 95.0 / 93.0 / 85.4 |

By the brief's marks:

- **No arm moves accuracy at 40 letters.** The best is 12%, against a ceiling of 82–85%.
- **At 20 letters H and K are partial gains.** Nothing moves or closes.
- **The strongest push did hold keepers:** 55–80 keepers that write at 12 letters and 36–41 at 20, against bars of 25–31. Accuracy at 20 letters did not follow, so the brief's sentence applies: these keepers, held from update 4,096, do not give accuracy at that distance.
- **The two weaker pushes did not hold keepers.** The 0.05 push ended with 16–21 at 12 letters, the 0.01 push with 1–3.
- **No arm harms 12 letters.**
- **Four of twelve predictions hold:** P32.4, P32.6a, P32.9 and P32.10.

Read plainly, the keeper account did not survive its own test at 20 letters, and none of the tested switches explains the loss at 40.

##### E31

Worker A's nine lives are complete. Its third 0.05-push life came in at 78.2 on taught pairs and 45.1 on untaught, so that arm is 50.0 / 73.6 / 45.1 on untaught pairs: mixed. Worker A reports P31.1 misses, P31.2 holds and P31.7 misses; I have not checked those judgments.

The default branch still has not moved, so the coordinator has read none of this. I have not scheduled another check; say if you want one when the last six lives land.

## 15. 2026-11-15 02:38 UTC — Mr. Xu

> Discuss what we should do next. Should we wait?

## 16. 2026-11-15 02:41 UTC — Reviewer

**Don't wait.** The one thing still running is worker B's last six E31 lives, due about 04:50 UTC. They complete the push arm's dose series and will not change the main readings, so nothing below depends on them.

##### What the two experiments now say

- **E31, lives with an answer mask: the recurrent map's backward pass is the bottleneck, and learned feedback fixes it.** I checked these from the receipts.
  - H (exact weights, a bound) gains 19–34 points on untaught pairs and lands within 9 of the ceiling in all three seeds, so P31.2 holds.
  - G gains 15 or more in only one seed, so P31.1 misses.
  - K (learned feedback, the candidate) scores 74.4 / 78.2 / 63.9 against a floor of 54.2 / 44.5 / 37.2. P31.3 misses on one seed (7.5 points below H, limit 5), but by the brief's 15-point rule K is not distinguishable from H.
  - K also clears E26's original marks (85 on taught pairs, 40 on untaught) in all three seeds, one of them barely (85.8). No arm inside the line had done that before.
- **E32, births without an answer mask: nothing tested fixes it.** No arm moves 40 letters. The strongest push held its keepers and accuracy did not follow, so the keeper account I proposed failed its own test at 20 letters.
- **Worker B's machine reset at 00:45 UTC** and killed three lives. It restarted them itself and checks in hourly; the cost was about an hour.

##### What I would do next

1. **Have the coordinator read and merge.** The default branch has not moved in 26 hours, and E32 and worker A's nine lives are complete.
2. **Replicate K on fresh seeds before the close on the 17th.**
   - Three new seeds: a sweep birth each, then a floor life, a K life and a backprop life from each.
   - E26's marks, registered beforehand.
   - About 15 CPU-hours, five to six hours on one machine.
   - Every piece is code that already ran in E26, E26-X or E31, so it needs no second implementation and can start as soon as it is written down.
3. **Localize the every-byte loss before trying another remedy.** Two small rosters of births, about 19 CPU-hours together:
   - **By crossing:** exact weights at the readout only; then the readout and the map into the hidden layer; then those plus the recurrent map; then everything except the recurrent map. These are bounds, 12 births, and need new code written twice as before.
   - **By target:** the sweep with the loss on the answer plus one other kind of byte at a time (structure, operands, letters). On file, answer-only holds 96–100% at 40 letters and every-byte holds 2%. This is a diagnostic of 9 births and needs only a loss mask in E28's runner.
4. **Keep the target test on hold.** It has no answer mask, and that condition is still unsolved.

##### Decisions for you

- **Ks.** Worker B's queue ends at about 04:50 UTC and Ks would be next. I would drop it for now: K already matches H, and the replication is a better use of that machine.
- **Who writes steps 2 and 3.** I can write both as specifications tonight for the coordinator to adopt. The catch is that I would then be auditing my own design; for step 2 that risk is small because it has no new code.

## 17. 2026-11-15 02:49 UTC — Mr. Xu

> I am not sure whether the replicate is worth the CPU powers. What is your reasoning?

## 18. 2026-11-15 02:50 UTC — Reviewer

My reasoning was weaker than I made it sound, and I think your doubt is right: I would not run it as a separate roster now.

**Why I proposed it:**
- K is the first arm inside the line to clear E26's marks, and it does so on single lives.
- The three seeds have been reused for every design choice since E20.
- The briefs themselves say a result is run again on fresh seeds before a claim.

**Why that does not justify 15 CPU-hours today:**
- **The finding that matters is already well supported.** K and H are each 19 to 34 points above their own floor on untaught pairs, in six lives out of six, paired by birth. The briefs' no-read margin is 15 points. A rerun would most likely confirm it, so it buys little information.
- **The fragile part is the mark, not the effect.** On taught pairs K gains only 5 to 9 points, which is inside the no-read margin, and one seed clears the 85 mark by 0.8. A replication would mostly test whether that seed lands on the right side of a threshold.
- **It does not move the blocker.** These lives all use the answer mask. The target test has none, and E32 showed that condition is still unsolved. Hours spent replicating K are not spent on that.

**What I would do instead:**
1. Say in the close exactly what is on file: K recovers the gap on untaught pairs in three of three seeds, it clears the E26 marks with one seed at 85.8, and it is not yet replicated.
2. Put the machines on the every-byte question. The by-target roster goes first: nine births, about 8 CPU-hours, one loss mask in a runner that already exists. Worker A's machine is idle now.
3. Fold fresh seeds into the next experiment on lives that is run for its own sake, with a floor arm beside it. Replication then comes as a by-product.

The one case for running it now is if the close on the 17th has to say more than "candidate on three seeds". Then a cut-down version is enough: three fresh births with a floor life and a K life each, about 10 CPU-hours.

## 19. 2026-11-15 02:51 UTC — Mr. Xu

> The writing above is poor. Use standard academic language, put a bit more thought and update what you think we need to do next. A simple question got you back and forth already.

## 20. 2026-11-15 02:53 UTC — Reviewer

I revised my recommendation on replication between two consecutive answers. The position below is the one I hold, with the reasoning set out once.

##### 1. What the evidence now establishes

Two results from the last 24 hours bear on the decision.

**Compositional generalisation in the continual-learning phase (E31).** Under sign-symmetric feedback, accuracy on held-out operator pairs was 54.2 / 44.5 / 37.2% across the three seeds, against 72.3 / 82.8 / 73.9% under backpropagation through time (BPTT). Two interventions on the recurrent matrix alone recover most of that deficit:

| Condition | Held-out pairs (%) | Gain over baseline (points) |
|---|---|---|
| Exact transposed recurrent weights (upper bound) | 72.9 / 74.5 / 71.4 | +18.7 / +30.0 / +34.2 |
| Kolen–Pollack learned feedback (admissible candidate) | 74.4 / 78.2 / 63.9 | +20.2 / +33.7 / +26.7 |

All six paired gains exceed the pre-registered 15-point margin. The deficit is therefore attributable chiefly to the sign approximation of the recurrent Jacobian, and a local rule repairs it.

**Length extrapolation under a full-sequence loss (E32).** When every byte is a prediction target, networks pretrained with sign-symmetric feedback fail beyond the trained range of 0–12 distractors: 1–2% at 40 distractors, against 82–85% under BPTT. No intervention changed this; the best condition reached 12%. The strongest gate-saturation penalty preserved the persistent units (55–80 against a criterion of 25–31), yet accuracy at 20 distractors did not follow. The hypothesis I advanced, that loss of persistent units causes the failure, is not supported.

##### 2. The replication question

A dedicated replication of the Kolen–Pollack result is not justified at present, for three reasons.

1. **The effect is large and consistent.** Six of six paired gains lie between 19 and 34 points, so a rerun on new seeds would very probably confirm it and yield little information.
2. **The fragile element is a threshold, not the effect.** On in-distribution compositions the gain is only 5–9 points, and one seed passes the 85% criterion at 85.8%. A replication would mainly test which side of that threshold a single run falls on.
3. **It does not address the binding constraint.** The target test has no answer mask, and E32 shows that this condition remains unsolved.

The result should be reported as a candidate on three seeds, not yet replicated. Replication belongs in the confirmatory run of whichever rule first passes the full-sequence condition, since that run needs new seeds in any case.

##### 3. Proposed next experiment

The two regimes differ in one respect that E32 did not manipulate. With an answer-only loss, sign-symmetric feedback extrapolates at 96–100%; with error also arriving from non-answer positions it does not, and reweighting those positions to half the loss does not help (2–16%). The open question is which class of non-answer target is responsible:

- distractor letters, whose entropy is irreducible;
- operand digits, also irreducible;
- structural bytes, which are predictable.

**Design.** Sign-symmetric pretraining with the balanced loss, varying only the set of non-answer targets:

- three conditions that add one class to the answer;
- three that omit one class.

The existing runner already labels each position by class, so no change to the backward pass is required. The reference cells are on file.

**Staging.**
- **Stage 1:** all six conditions on one seed, about 6 CPU-hours. The contrast of interest is roughly 2% against 96%, so a single seed suffices for screening.
- **Stage 2:** only the conditions that prove decisive, on the remaining two seeds.

**Decision rule.**
- **If removing one class restores extrapolation,** the cause is interference from that class under approximate feedback. The next candidate is then a local, uncertainty-weighted loss, which needs no knowledge of the answer's position.
- **If no single class accounts for it,** the remedy lies in the feedback pathway. The next roster is then exact feedback at the readout-side maps as an upper bound, and sign-initialised Kolen–Pollack feedback on all maps as the candidate. This needs new code and two independent implementations.

##### 4. Other recommendations

- **Consolidate the record.** The default branch has not advanced in over 26 hours; E32 and nine E31 runs await reading and merging.
- **Do not run the sign-initialised variant (Ks) in the continual phase.** The standard variant already matches its upper bound.
- **Do not pursue further gate-saturation variants.** They were insufficient in E32 and mixed in E31.
- **Keep the target test on hold** until one rule passes the full-sequence condition.

##### 5. Decisions requested

1. Approval of Stage 1 (six runs, one seed).
2. Whether I should draft its specification for the coordinator to adopt. I would then be reviewing my own design, though the risk is limited because it adds only a loss mask.

## 21. 2026-11-15 03:06 UTC — Mr. Xu

> Message for the reviewer: "Please audit `OPUS_TASK_E33_WHICH_TARGETS_BREAK_EXTRAPOLATION_20261115.md` on the default branch (`codex/minifly-consolidated-20261023`, commit 00fad472) before any worker starts. The design is yours; the specification is mine. Give each finding with the command that shows it. In particular check:
>
> 1. whether the loss code block runs against the real `exp28_everybyte.masks` and `FINE`, and whether `A+DOS` is exactly S-bal's loss and `A` is the answer-only loss at half weight;
> 2. whether my weighting choice (kept positions keep S-bal's weight, dropped positions are not compensated) is the right one for your question, or whether rescaling is needed, and whether the 'kind against amount' control in stage 2 is sound;
> 3. whether adding the two end arms is right, and whether stopping on outcome 0 is the correct response if they miss the references;
> 4. whether the three classes are grouped correctly, given that the task code inside 'structural' is itself unpredictable;
> 5. whether the marks (learned in range at 80, extrapolates at 60, fails at 20, mean of the last three rows) are computable and sensible on one seed, using the rows on file for S-bal and the E32 arms;
> 6. whether the outcome table covers every pattern of eight arms and whether any reading claims more than one seed can show;
> 7. whether the checks are achievable as written, in particular the S-bal hash check at 2,048 updates and the zero-gradient check at the logits;
> 8. whether `e33_read.py` can import `panels`, `measure`, `load_net` and `row_files` from `e32_read.py` as it stands on the default branch.
>
> Say plainly at the end whether the brief can go to a worker as it stands, or name the smallest set of changes that would let it."
> Paste the findings here when they come back. No worker starts on E33 until then.

## 22. 2026-11-15 03:12 UTC — Reviewer

#### Audit of the E33 brief (commit `00fad472`): three text changes required, then it can go to a worker

The specification is faithful to the design and its code is sound. Three items need changing before a worker starts, all in the text. Because the design is mine, two of the findings below concern my design as much as your specification.

The supporting script is on `claude/vibrant-cerf-sje726` at `7df89744`:

```
D=origin/codex/minifly-consolidated-20261023; M=MOD97_LOCAL_RULE_20261106; R=PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109
python3 -I $R/e33_brief_checks.py
```

##### 1. The loss code block

The block is consistent with the real functions.

- `masks` returns the five values in the order used, and `FINE` contains the five role names.
  `git show $D:$M/exp28_everybyte.py | sed -n '63p;113,131p'`
- The assertion holds by construction: `roles_of` assigns every non-answer target one of the five roles (lines 97–110).
- **`A+DOS` is S-bal's loss.** I expect it to be bit-identical, because multiplying by 0.5 is exact in binary floating point. I could not run torch here; the brief's fallback covers the other case.
- **`A` is the answer-only loss at half weight,** as stated. It cannot be compared with E26's birth by hash. Under AdamW a constant factor should alter the update only through the optimiser's epsilon, so I expect the same behaviour; that is untested.
- Minor: the block mixes `torch.` and `T.`.

##### 2. The weighting choice and the control

Your choice is the right one, and no rescaling is needed.

- **Why it is right.** Each arm is S-bal with terms deleted. In the leave-one-out arms the remaining terms keep exactly their S-bal weights, which is what a test of necessity requires. Rescaling would hold total weight fixed but inflate the error of the retained class, replacing one confound with another.
- **Two outcome patterns coincide with a pure amount effect.** The brief names this confound only as class size.

  | Arm | Non-answer weight | Error per item (nats) |
  |---|---|---|
  | A+S | 0.115 | 1.4 |
  | A+O | 0.154 | 10.8 |
  | A+D | 0.231 | 20.6 |
  | A+OS | 0.269 | 12.2 |
  | A+DS | 0.346 | 22.0 |
  | A+DO | 0.385 | 31.4 |
  | A+DOS | 0.500 | 32.8 |

  - Outcome 1 with D decisive is exactly what a threshold on error between 12.2 and 20.6 nats would produce.
  - Outcome 3 with all pairs failing is exactly what a threshold on weight between 0.231 and 0.269 would produce.
- **The control is sound for D.** Halving D's terms leaves 10.2 nats per item, about what O carries. Failure then indicates kind; extrapolation indicates amount.
- **The control has no fixed reading.** State it now, in terms of error per item.
- **For O or S the control is unnecessary.** If either were decisive, a class carrying less error would fail while D passes, which already excludes amount.

##### 3. The two end arms and outcome 0

Adding them is right. S-bal and the answer-only birth have only their final files on the default branch, so these arms supply the rows the endpoint needs.

- **Stopping on outcome 0 is correct, but its stated reason is wrong in one case.** If check 1 passes, `A+DOS` is S-bal. Its missing the "fails" mark would then mean the mark is mis-set, not that the runner is faulty. S-bal's last-three-row mean at 40 letters is not known; only its final row (6.0) is.
- **Add a check.** S-bal's receipt records a hash for all six rows. If the dry birth is bit-equal, the full `A+DOS` birth should match every one.

##### 4. The three classes

The grouping is acceptable and need not change.

- **Class S is, in effect, the task code alone.** At the end of S-bal the loss per position is 1.40 nats for the code, 0.004 for the operator and 0.000 for `=`.
- **The code is an unusual target.** It is unpredictable but has no history: it sits at position 0, where the state is zero.
- **Consequence for the readings.** "S is harmless" would show that one history-free unpredictable target is harmless. It would show nothing about predictable targets, which carry no error once learned in any arm.

##### 5. The marks

They are computable for the eight arms. Two thresholds are tight.

- **"Learned in range" at 80 leaves little margin.** On the receipts' own panel S-bal's last-three-row mean at 12 letters is 86.7 / 90.3 / 85.3, and its single rows swing by up to 12 points. One arm at 78 would turn a clean pattern into "mixed". A bar of 70 would be safer; this is a recommendation, not a requirement.
- **"Fails" at 20 is adequate for seed 73129** (S-bal's final row is 6.0). In stage 2 it is close for seed 93683 (16.3).
- **"Extrapolates" at 60 is sensible:** the BPTT ceiling is 82.0 and the answer-only birth 96.0.
- **The endpoint cannot be independently recomputed** if only final files are committed. Committing rows 16,384 and 20,480 for each arm (16 files, about 72 MB) would allow it. Also a recommendation.

##### 6. The outcome table

Row 4 makes the table complete, but it absorbs patterns that deserve a reading.

- **Six of eighteen clean patterns fall into "mixed".** I enumerated every pattern in which adding targets never helps. The six are the three "either of two classes suffices" patterns and the three "one class, or the other two together" patterns.
- **One of them is the pattern my own hypothesis predicts:** D or O suffices and S is harmless.
- **Fix.** Replace rows 1 and 2 with one rule: the failing arms are exactly those containing at least one class of a set C, read for C of size one, two or three.
- **Two readings need a qualifier.**
  - Outcome 1 with D decisive should say "pending the control".
  - Outcome 3 with all pairs failing should say it is what a weight threshold predicts.

##### 7. The checks

All are achievable as written.

- **Hash at 2,048 updates:** the reference exists with its `tensors_sha256`, and E32's runner showed that a rewritten loop can reproduce E28's dry hash.
- **Zero gradient at the logits:** the logits are a non-leaf tensor, so `retain_grad` applies, and a loss over indexed positions gives exact zeros elsewhere.
- **Data digest:** the field exists at every row.

##### 8. The reader

`e33_read.py` can import all four functions as they stand. They are top-level and the module has a main guard.

`git show $D:$M/e32_read.py | grep -n '^def \|__main__'`

One dependency: `row_files` requires a receipt with `complete` true and exactly six entries in `row_weights`, the last at 24,576. E33's receipts must keep E28's format, which the brief's loop does.

##### Verdict

Not as it stands. Required, all text:

1. Generalise outcome rows 1 and 2 to "fails if and only if the targets include a class in C", and add the qualifiers in point 6.
2. Reword outcome 0, and add the six-row hash check for `A+DOS`.
3. State both amount confounds with the table in point 2, and fix the control's reading in terms of error per item.

Recommended only: a bar of 70 for "learned in range"; committing two more rows per arm; describing S as the task code in effect.

With the three required changes made as written, I do not need to see the brief again.

##### Not checked

- I ran no torch code, so the bit-equality in point 1 is argued, not run.
- Stage 2 beyond the control.
- The cost figure, beyond S-bal's recorded 2,867 CPU-seconds.

## 23. 2026-11-15 16:05 UTC — Mr. Xu

> Discuss next steps.

## 24. 2026-11-15 16:10 UTC — Reviewer

##### Summary

E33 has identified where the full-sequence failure enters. I recommend a reduced replication (four births) and a first test of a specific remedy (three births), about six CPU-hours in total, both of which can start today. One ruling from you is required for the remedy.

##### 1. Evidence since the last discussion

**E33, stage 1 (seed 73129).** Targets at the distractor positions are what prevent length extrapolation under sign-symmetric feedback. My reader reproduces all eight endpoints exactly.

| Distractor targets | Arms | Accuracy at 40 distractors (%) |
|---|---|---|
| absent | A, A+O, A+S, A+OS | 94.8, 90.2, 88.1, 94.2 |
| present | A+D, A+DS, A+DO, A+DOS | 23.8, 38.1, 2.3, 4.1 |

- **Formal outcome.** By the registered rule it is "mixed", because two arms fall between the marks, so no class is formally named. The contrast is nevertheless unambiguous on this seed: 17.1% against 91.8%.
- **Other classes.** Operand targets worsen the result only when distractor targets are present; the task code has no effect.
- **Amount versus kind.** A weight threshold is excluded: A+OS carries more non-answer weight than A+D (0.269 against 0.231) and extrapolates. A threshold on error is not yet excluded.
- **Upper bound.** Any remedy acting on the loss can reach at most about 94% (A+OS).

**E31, complete.** The gate-saturation penalty did not improve composition at any strength (changes of −11 to +29 points, with no strength helping in two seeds). Together with E32, this closes that line.

##### 2. Proposed programme

**Step 1. Reduced stage 2 of E33.** Run A+D and A+OS on seeds 93683 and 116089: four births, about 3 CPU-hours. The brief requires replication before any claim, and these two arms carry the contrast.

**Step 2. A candidate remedy (E34), concurrently.** Scale each position's output error by a function of the network's own predictive entropy at that position, w = exp(−H/τ). The rule uses no label and no knowledge of where the answer is.

- **Feasibility.** I measured entropy by target class on the saved rows of the baseline. It separates the classes throughout training: about 3.46 nats at distractor targets against 0.3–0.7 at answer targets.
- **Expected effect of the weighting at τ = 1.** These are projections from networks trained without it.

  | | Unweighted | Weighted |
  |---|---|---|
  | Distractors' share of total error | 60–61% | about 30% |
  | Answer's share of total error | 1–5% | 12–16% |

- **Design.** Sign-symmetric pretraining with every byte a target, at three temperatures on one seed (three births). A temperature that passes then runs on the other two seeds. The references are on file: 2% baseline, 94% bound, 82–85% under BPTT.
- **Dose-response.** The three temperatures span a wide range of distractor weight, so no separate dose series is needed.
- **Built-in check.** With unit weights the birth must reproduce the baseline's recorded hash.

**Step 3. Conditional on E34.**

- **If a temperature extrapolates:** a confirmatory experiment on new seeds, without the answer mask, combining the weighting with Kolen–Pollack feedback in the continual phase. This also replicates the Kolen–Pollack result.
- **If none does:** the feedback pathway is next, with Kolen–Pollack feedback on all maps as the candidate. That requires new code and two independent implementations.

**Step 4. Consolidation.** E33 stage 1 and the final E31 report are not yet on the default branch. The close on 17 November can state: parity at pretraining; composition recovered by learned recurrent feedback on three seeds, unreplicated; the full-sequence failure localised to distractor targets on one seed.

##### 3. Not recommended

- Further gate-saturation variants.
- The sign-initialised Kolen–Pollack variant.
- A dedicated replication of the Kolen–Pollack result.
- The target test, until one rule passes the full-sequence condition.

##### 4. Decisions requested

1. Release of the reduced stage 2 (four births).
2. A ruling on admissibility: whether gain modulation of the output error by the network's own predictive entropy lies inside the line.
3. Whether I should draft the E34 specification for the coordinator to adopt.

## 25. 2026-11-15 16:24 UTC — Mr. Xu

> Both deliveries are merged and E33's endpoints recompute from its read file. I'm now measuring predictive entropy by target class on the saved baseline rows, so the E34 brief rests on figures I've checked.
> Entropy figures reproduce the reviewer's (distractor share 60% falling to about 30% at τ = 1). One last check on how small the weights are at the start of a birth, then I write both briefs.
> Checks are done and filed. Writing the two briefs now.
> Both briefs are written and pushed (commit 9291d23b, confirmed on the remote). The reduced E33 stage 2 is released; the E34 brief is drafted and waits for your ruling and the reviewer's audit.
> What I did first
>
> * Merged E33 stage 1 and the complete E31 report into the default branch.
> * E33 stage 1: the eight endpoints recompute from the worker's read file and match the report. I did not re-measure the weight files; the reviewer says its reader reproduces them.
> * E31 correction: I had recorded the weakest gate push (P2c) as not run. Worker B ran its three lives, and they are in the table; the review now says so.
>
> E33 addendum 1: reduced stage 2 (released)
> Arms `A+D` and `A+OS` on seeds 93683 and 116089: four births, about 3.2 CPU-hours. The addendum fixes, before the run:
>
> * Replication rule: on a new seed, `A+OS` must extrapolate and `A+D` must not, staying at least 40 points below it at 40 letters.
> * Sentence it licenses: only that the contrast holds on three seeds, with nothing about why or about amount against kind.
> * Predictions: P33.8 to P33.11.
>
> E34 brief (not released)
> The rule is the reviewer's: each byte's error is multiplied by exp(−H/τ), where H is the network's own predictive entropy at that byte. It uses no label and no answer mask. The design is three temperatures (0.5, 1, 2) on seed 73129, plus a unit-weight dry birth that must reproduce the baseline's hash.
> What I measured before writing it (scripts and output are in `PROJECT_DOCUMENTS/COORD_CHECKS_20261115/`):
>
> * The reviewer's entropy figures reproduce. Distractors sit at 3.45–3.48 nats and answer digits fall from 0.98 to 0.42. Distractors' share of the error is about 60% unweighted and about 30% at τ = 1.
> * The temperatures span the dose. At τ = 2 a distractor byte gets 0.20–0.23 of an answer byte's weight, about what `A+D` had (0.245). At τ = 1 it gets 0.04–0.05, and at τ = 0.5 about 0.002.
> * The rule as outlined would stall at the start. On an untrained network every weight is tiny, and at τ = 0.5 the median gradient (1.7e-9) is below the optimizer's floor (1e-8). I added a division by the sum of the weights, which restores the baseline's gradient scale and keeps the unit-weight loss bit-equal to the baseline's. That division is one number shared across the batch.
> * The baseline misses the "learned in range" mark. Its last three rows at 12 letters are 66, 37 and 75, so an arm that learns in range is already an improvement.
>
> Your ruling
> The question has two parts, and the brief says what happens under each answer:
>
> 1. May each output position scale its own error by a gain computed from its own predicted distribution?
> 2. May all positions in a batch share one further common gain (the division above)?
>
> If you rule the second part outside, I revise the brief before release.
> Messages to relay
> E33 worker (same worker as stage 1): "Pull the default branch and do addendum 1 of `OPUS_TASK_E33_WHICH_TARGETS_BREAK_EXTRAPOLATION_20261115.md`: `A+D` and `A+OS` on seeds 93683 and 116089. Add the seed argument to the reader and do the two checks first. Launch so that no wrapper limit under 90 minutes applies. Report the per-seed table, which of R1 to R4 applies on each seed, and P33.8 to P33.11, with no readings beyond the addendum's."
> Reviewer: "Please audit `OPUS_TASK_E34_ENTROPY_WEIGHTED_ERROR_20261115.md` at 9291d23b, and addendum 1 of the E33 brief if you wish. Points to check:
>
> 1. Is the loss code your rule, and is the division by `w.sum()` acceptable or does it change what you meant to test?
> 2. Do my entropy and weight-ratio figures agree with yours (`PROJECT_DOCUMENTS/COORD_CHECKS_20261115/`)?
> 3. Are τ = 0.5, 1, 2 the right three?
> 4. Is the outcome table complete, and are the marks right given that the baseline does not learn in range?
> 5. Are the five checks sufficient and computable?
> 6. Is the stage 2 selection rule sound?"
>
> Standing
>
> * Held: the target test, E29 part B, further gate pushes, and a dedicated replication of the Kolen–Pollack result.
> * Close: the one-page arithmetic close is still due to you on the 16th with open items marked.

## 26. 2026-11-15 16:29 UTC — Reviewer

#### Audit of the E34 brief (commit `9291d23b`): sound; one required change to the stage 2 rule

The specification implements the rule I proposed, and the coordinator's measurements agree with mine. One change is required before release, in the stage 2 selection rule; four smaller ones are recommended. Release also still depends on your ruling on admissibility.

The supporting script is on `claude/vibrant-cerf-sje726` at `12eb5a9f`:

```
R=PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109
python3 -I $R/e33_brief_checks.py e34
```

##### 1. The loss code and the division by `w.sum()`

The code is the rule I meant: the error at each position's logits is (softmax − one-hot) scaled by a detached weight exp(−H/τ).

The division is acceptable, and I consider it the better form.

- **It leaves relative weights unchanged.** It is one scalar per batch, so it does not alter what is tested.
- **It removes an optimiser artefact.** The coordinator's first-batch figures are arithmetically consistent: the baseline's median gradient of 1.07e-4 times an initial weight of 1.56e-5 gives 1.67e-9, below the optimiser's floor of 1e-8. I did not rerun them; torch is absent here.
- **It does not hold the gradient scale constant later.** As confident positions with near-zero error come to dominate the sum, I estimate the scale at τ = 0.5 falls to roughly a twentieth of the baseline's. That is still several hundred times the floor.
- **For the ruling.** The sum over a batch is a global scalar. If you rule it outside, a fixed constant would work, but the loss scale would then vary by orders of magnitude during training, which the normalised form avoids.

Recommended: add `assert w.sum() > 0`.

##### 2. Entropy and weight-ratio figures

They agree with mine. The table gives the weight of a distractor position relative to an answer position.

| Row | τ | Mine | Coordinator's |
|---|---|---|---|
| 4,096 | 0.5 / 1 / 2 | 0.0019 / 0.0489 / 0.2320 | 0.0019 / 0.0489 / 0.2320 |
| 24,576 | 0.5 / 1 / 2 | 0.0014 / 0.0394 / 0.2003 | 0.0014 / 0.0393 / 0.2001 |

- **Shares of the weighted error** agree within about one percentage point, and the entropies by class to two decimals.
- **The comparison with E33's `A+D`** (0.245) is correct: 192 answer positions against 782 others.

One qualification is missing from the brief. The projections begin at row 4,096, when the answer is partly learned. Earlier the gain is weaker. From the baseline's own loss record, if entropy tracks loss, the ratio over the first 512 updates is about 0.08, 0.29 and 0.54 at τ = 0.5, 1 and 2. At τ = 1 the rule therefore starts above `A+D`'s dose and falls below it within about 1,000 updates.

Recommended: have the report give the realised ratio for the first eight blocks, not only at the end.

##### 3. The three temperatures

They are the right three for a first screen.

- **τ = 2** reproduces `A+D`'s dose and should fail, which ties the rule to E33.
- **τ = 1 and τ = 0.5** span a fifth to a hundredth of that dose on a logarithmic scale.
- **Nothing below 0.5.** At τ = 0.25 an unlearned answer digit would carry a weight near 1e-4, and I would expect its learning to be starved.

##### 4. Outcome table and marks

The table is complete. Each arm is in exactly one of four states (not learned, extrapolates, intermediate, fails), and rows 1 to 4 exhaust the combinations in order.

The marks are appropriate. The baseline's failure to learn in range is handled by reporting each arm's 12-letter accuracy beside it.

Recommended:

- **Cite the like-for-like reference as a last-three-row mean.** For a rule without a mask the reference is BPTT with every byte a target. E32's rerun gives 92.8 at 12 letters and 82.0 at 40 for this seed; the brief cites final rows.
- **Report whether a passing arm is within 15 points of 82.0.** The "extrapolates" mark of 60 is 22 points below that reference.

##### 5. The five checks

They are sufficient and computable. The dry reference for the baseline exists and was matched by E32's runner; the logits are a non-leaf tensor, so check 2 is feasible. When the rows are committed I will recompute entropy by class with my own reader against the receipts' diagnostics.

##### 6. The stage 2 selection rule

This is the required change.

- **The defect.** The rule selects the passing arm with the highest accuracy at 40 letters. Two passing arms will usually differ by less than 15 points, which the programme's own rule treats as unreadable. The selection would therefore be made on noise, and on a seed that then counts towards "three seeds".
- **Fix, either of two.**
  - Run every passing arm on the two new seeds, at about 1.6 CPU-hours per arm, and state the sentence per temperature.
  - Or treat differences under 15 points as ties, with ties going to the larger τ.
- **In both cases** the sentence should say that one of the three seeds selected the temperature.

##### E33 addendum 1

It is sound. One small gap: rows R2 and R3 can both apply, so the order of precedence should be stated.

##### Verdict

The brief can go to a worker once the ruling is given and the stage 2 rule in point 6 is changed. With that made as written, I do not need to see it again.

##### Not checked

- I ran no torch code.
- The first-batch figures in point 1 are checked by arithmetic only.
- The estimate of the later gradient scale is mine and rough.

##### One operational note

The E33 worker's branch has not moved since its stage 1 report at 06:39 UTC, so the stage 2 message appears not to have reached it. I can deliver it if you wish.

## 27. 2026-11-15 16:55 UTC — Mr. Xu

> 现在项目在什么状态呢？

## 28. 2026-11-15 16:56 UTC — Reviewer

截至 11 月 15 日 16:55（UTC），项目处于"原因已基本定位、第一个补救规则刚获准并开始实验"的阶段。

##### 已确立的结果

除注明外均为三颗种子。

1. **预训练阶段（仅以答案为目标）**：符号对称反馈与时间反向传播（BPTT）持平，准确率 98–100%。
2. **持续学习阶段的组合泛化**（未教过的运算对）：
   - 符号对称反馈为 37–54%，BPTT 为 72–83%。
   - 仅在循环矩阵上改用精确权重（上界）可达 71–75%。
   - 改用 Kolen–Pollack 学习反馈（界内的候选规则）可达 64–78%，尚未在新种子上复现。
   - 门控饱和惩罚在任何强度下都无效，这条路线已放弃。
3. **无答案掩码时的长度外推**（训练 0–12 个干扰字母，测试 40 个）：
   - 符号对称反馈为 1–2%，BPTT 为 82–85%。
   - E32 表明，改动循环反馈通路、门控行或强行保留"保持单元"都不能改善。我提出的保持单元假说在 20 个字母处未通过它自己的检验。
4. **E33 第一阶段（仅一颗种子）**：失败来自干扰字母位置上的预测目标。
   - 不含这类目标的四个条件为 88–95%，含这类目标的四个条件为 2–38%。
   - 按预先登记的规则，结果记为"混合"，不正式指认某一类。

##### 正在进行

- **E33 第二阶段（精简版）**：在另外两颗种子上重复关键对比，共 4 次预训练，约 3.2 CPU 小时。工作会话在 16:53 时处于运行状态。
- **E34（熵加权误差）**：每个位置的输出误差按网络自身预测熵缩放，不使用任何标签或答案位置。
  - 您已裁定该规则在界内。
  - 简报第 2 版已于 16:51 发布，采纳了我的一项必改意见和四项建议。
  - 设计为三个温度（0.5、1、2）在一颗种子上筛选，约 2.5 CPU 小时。
  - 我尚未看到任何 E34 的工作分支活动，消息是否已送达工作会话需要确认。
- **17 日的收尾**：协调者的一页总结定于 16 日交给您。

##### 接下来的判断点

- **若 E34 有温度通过**（12 个字母处不低于 70%，40 个字母处不低于 60%）：先在另外两颗种子上验证，然后在新种子上做确认实验，并与 Kolen–Pollack 反馈结合进入持续学习阶段。
- **若三个温度都不通过**：转向反馈通路，以全部映射上的 Kolen–Pollack 反馈为候选。这需要新代码和两份独立实现。

##### 暂缓事项

目标测试、对 Kolen–Pollack 结果的单独复现、门控类变体。

##### 需要留意

- 关键结论目前只建立在三颗种子上，E33 的定位只有一颗。
- 我与协调者属于同一模型家族，我的审计只是第二读者，不是独立审计。

## 29. 2026-11-15 16:57 UTC — Mr. Xu

> 我不是非常理解，我们反复和backpropagation进行比较，固然能够帮助发现问题，但是似乎无法找到超越backpropagation的机制/模型。你对此怎么看？

## 30. 2026-11-15 16:58 UTC — Reviewer

您的判断是对的：按目前的实验设计，这条路线最多只能在局部性约束下追平反向传播，不可能超越它。

##### 一、为什么会这样

- **上限是按构造设定的。** 每个实验都把 BPTT 当作"上限"，所有补救措施都是对它的逼近。Kolen–Pollack 反馈在数学上就收敛到精确的转置权重，也就是收敛到反向传播本身。
- **此前这样做有其必要。** 它起的是阳性对照的作用：没有一个已知可行的参照，就无法区分"规则的失败"和"任务或数据的失败"。两处定位（循环矩阵上的信用分配、干扰字母目标）都是靠它在几天内得到的。
- **但对照不是目的地。** 反向传播计算的是给定损失的精确梯度；在同一任务、同一数据、同一网络、同一损失上，任何梯度的近似都不会比它下降得更好。

##### 二、"超越"只可能出现在哪里

要胜过反向传播，改变的必须是梯度计算以外的东西。我们自己的数据已经给出三类线索。

1. **学什么，而不是怎么算梯度。**
   - BPTT 在全字节目标下的外推只有 82–85%，仅以答案为目标时为 92–99%，可见干扰目标同样损害反向传播。
   - E33 中去掉干扰目标的条件达到 94%（一颗种子）。
   - 因此 E34 的熵加权若成立，就可能高于通常做法下的 BPTT。这是目标函数层面的改进，与信用分配规则无关。
2. **记忆与结构。**
   - 无重放的持续学习中，两种规则在难题对上都跌到约 1%，这是反向传播的结构性弱点。
   - 快速绑定与修订需要一次呈现即学会，梯度下降做不到。
   - 先天电路（E10 的环形算术）组合准确率为 100%，而 BPTT 训练的通用网络为 72–83%。
3. **代价。** BPTT 需要存储整段历史并做全局的权重传递；局部规则在内存、能耗和硬件可实现性上可以更优。这是文献中"超越反向传播"最常见的含义。

##### 三、对项目的建议

1. **把反向传播从"上限"改为"基线"。** 每个指标另设理想上限，例如贝叶斯最优、读取真值的理想模型或手工布线的解。这样才看得见基线之上还有多少空间。
2. **预先选定主战场。** 我建议三项：无重放的保持、少次呈现的绑定与修订、无掩码条件下的外推。前两项正是目标测试要测的内容。
3. **两个可以立即做的小改动。**
   - E34 增加"BPTT 加同样的熵加权"对照，以判断这一增益是否与规则无关。
   - 目标测试中反向传播的基线和理想上限现在就可以测，不必等局部规则通过全字节条件。我此前建议整体暂缓目标测试，这一部分应当放开。

##### 四、需要坦率说明的限度

- **规模的限制。** 在 256 单元、模运算这样的规模上，很难在"同任务最终精度"上得到学界会认可的超越。现实的贡献有两类：一组在无掩码、持续学习条件下追平反向传播的局部规则；以及在反向传播的弱项上胜过它的机制。
- **先天结构的取舍。** 您此前表示可以放弃果蝇式的先天结构。但先天结构恰恰是已知能在样本效率上胜过通用梯度学习的主要途径之一。若坚持通用网络从零学习，规则就是在反向传播的主场上与它竞争。这一取舍值得重新考虑。

## 31. 2026-11-15 17:00 UTC — Mr. Xu

> 如果你需要更多信息才能做判断，给deep research agent提供一个prompt。如果不需要，进一步讨论这个问题，因为我们已经做了很多实验有大量数据了。不要贸然下结论。

## 32. 2026-11-15 17:01 UTC — Mr. Xu

> 另外，考虑到将问题变难极为容易，第四个困难显然不是本质的。

## 33. 2026-11-15 17:02 UTC — Mr. Xu

> 此前实验发现了果蝇的结构似乎效果不好，复盘一下此前的决策过程。

## 34. 2026-11-15 17:05 UTC — Reviewer

先回应两点。规模那一条我撤回：把任务加难很容易，它不是本质限制。另外，这次复盘不需要外部检索，项目记录本身已经足够；我重读了 11 月 6 日的总评、主实验报告（E45）和我 9 日的备忘录。下文数字除注明外取自这些记录，E10 和中央复合体的结果我没有独立复核。

##### 一、"果蝇结构"在记录里指三样不同的东西

| 对象 | 做过的检验 | 结果 | 这个证据能说明什么 |
|---|---|---|---|
| A. 蘑菇体作为整个学习器（随机稀疏扩展加局部读出） | 约 220 个版本；10 月 17 日的顺序探针；11 月 6 日的 mod-97 实验 | 记忆很好，泛化为零。mod-97 上共享通路单独 86–92%，蘑菇体残差单独 0.9–1.6% | 较强：随机扩展编码把相似输入分开，按构造不利于泛化 |
| B. 蘑菇体作为快速记忆，放在共享学习器旁边 | 只有 11 月 8 日的一次消融（E16-A） | 去掉后结果差 0.00–0.04 个百分点 | 很弱：该存储从未进入损失，键在第一个区块就溢出，所有测试题又都是未见过的。它在那套实验里没有角色 |
| C. 先天计算回路（环形吸引子、中央复合体） | E10 先天环形运算；罗盘与转向回路；路径积分；计数环 | 见下 | 正面但范围窄，且依赖人为提供的结构 |

C 类的具体结果：

- **E10**：运算先天给定时，未教过的运算对零样本组合准确率按种子为 100、65、91，对照为 4–10%。准确率取决于接口学得多准；解析由实验者按位置提供。
- **罗盘与转向回路**：拟合 15 个增益后闭环成功 80/96，重连对照为 20–24。
- **路径积分**：未过关（返程残差 0.71，门槛 0.30）。记录写明原因是积分器由模型自己补上、没有取自连接组，结论是"除非积分器也取自连接组，否则关闭"。
- **计数环**：记录自己标为"无信息"，因为那是从 75 个样本做纯外推，任何学习器都过不了。

##### 二、决策时间线

| 日期 | 决定 | 依据 | 复盘 |
|---|---|---|---|
| 11 月 6 日 | 总评判定蘑菇体"只是联想记忆"。建议保留它作快速记忆，另加共享计算，并提出"加脑区，不是加大脑" | A 类证据 | 判断有据。注意它没有建议放弃果蝇结构 |
| 11 月 6–7 日 | 蘑菇体退出计算通路 | 残差约 1% | 依据的是蘑菇体单独的得分。我 9 日核过未合并分支上的审计：加上残差后的增量按种子为 +4.2、+2.5、−4.0，即没有可靠贡献。这一步站得住 |
| 11 月 7 日 | 计划第 3 步："先天运算的环，决定组合是否应当先天" | 无 | E10 当天给出了答案（先天则组合近乎免费），但此后没有任何决定接续它 |
| 11 月 8 日 | 存储"退役"，记录写"学习器受果蝇约束，但不具果蝇结构" | E16-A | 退役的理由是"在这套实验里不起作用"，不是"检验过并且失败"。记录自己写明，它该接受的检验（情景记忆，对照通用索引）从未做过 |
| 11 月 9 日 | 我把您的答复读作"果蝇结构不是必要条件" | 您关于"迷你"和重放代价的回答 | 这是我的解读，不是您的裁定 |
| 11 月 10 日 | 协调者决定先解决"凭记忆组合"，改用通用循环网络 | 无窗口学习器在 BPTT 下也不会组合 | 问题提得对。但它把主线带到通用网络上，先天回路一线从此无人过问。记录注明"此决定可由所有者否决" |
| 11 月 14 日 | 我把果蝇式架构称为"退路"，您答复可以放弃 | 我对机制目录的阅读 | 当时我没有区分 A、B、C，把三者当作一个东西提给您 |

##### 三、记录支持什么，不支持什么

- **支持**：蘑菇体式随机扩展不能单独充当会泛化的学习器。
- **不支持**："果蝇结构效果不好"这一笼统说法。B 类从未在需要它的任务上检验过；C 类的结果是正面的。
- **需要保留的疑虑**：
  - 11 月 4 日的备忘录指出，此前的成功大多依赖人为提供的结构。E10 把加法接成循环移位，等于把答案的一部分接进了线路。
  - E10 的操作数由解析直接给出，不是从字节流中凭记忆取得，因此不能与现在 GRU 的 72–83% 直接比较。
  - "先天运算是否越界"当时交给了所有者决定，我在记录里没有找到这项裁定。

过程上的问题有三个：

1. 三样东西共用一个名字，对 A 的否定被推广到了 B 和 C。
2. "退役"与"证伪"没有区分。
3. 正面结果（E10、罗盘回路）没有人接续。

其中第 1 条在 14 日和昨天都有我的责任。昨天我说"先天结构值得重新考虑"，依据是一般文献，没有先回到记录核对。

##### 四、现在还不能下的结论

"果蝇结构能否胜过反向传播"目前没有答案。记录里有两处现成的缺口可以判定它，都不需要新想法：

1. **B 类**：情景记忆与一次性绑定任务上，稀疏键存储对照通用索引，再对照重放。这正是反向传播的弱项，也是您提过的"果蝇不靠重放"的代价标准。
2. **C 类**：先请您裁定先天运算是否在界内。若在界内，把 E10 改为从字节流凭记忆取操作数，与 GRU 在同一任务上比较。

## 35. 2026-11-15 17:11 UTC — Mr. Xu

> 1） 但你没有回答超越backpropagation local learning mechanism是否需要deep research agent的问题。2）上面的果蝇结构是否能够充当其他框架的基础进行泛化学习的问题，讨论如何设计实验验证这个问题。注意到此前V223已经尝试了几百轮之多，效果都不好。3）只要不用全局的反向传播，任何机制都可以考虑。但需要做到泛化。

## 36. 2026-11-15 17:23 UTC — Reviewer

三点依次回答。在检索和实验结果回来之前，我不对"局部机制能否超越反向传播"下结论。

##### 一、是否需要 deep research：需要

1. **范围变宽了。** 您的第 3 点允许任何不含全局反向传播的机制，这远超我们试过的几种（符号对称、Kolen–Pollack 反馈、门控惩罚、熵加权、一次先天环）。
2. **我的记忆不足以作判断依据。** 文献中确有"在某些条件下优于反向传播"的主张，例如预测编码一类的能量模型、果蝇启发的持续学习、稀疏分布式记忆、向量符号结构。我只记得标题层面的结论，没有核对过条件和数据。
3. **检索只能提供候选，不能替代实验。** 上一轮检索给出的补救排序在我们的实验里没有应验：排第一的门控饱和代价无效，排最后的学习反馈矩阵有效。

提示词如下（英文，不含项目信息）：

```
I need a source-checked literature review to decide where to invest experiments. Open the primary papers, including figures, tables, supplements and code; do not rely on abstracts or secondary summaries. For every claim give the URL, the exact sentence, table or figure it rests on, the architecture and its size, the task and metric, what the baseline was and whether it was tuned, the number of seeds, peer-review status, and whether an independent group reproduced it. "Not found" and "could not verify" are acceptable answers; say what you searched.

Setting. Small recurrent networks (1e5 to 1e6 parameters, CPU budget) learn byte-level algorithmic streams: modular arithmetic with distractor symbols between the operands, and two-step compositions answered from memory. Evaluation is on held-out operand pairs, held-out operator combinations, and gaps longer than any seen in training.

Constraint. No end-to-end backpropagation of exact gradients through the whole network, in any phase. Everything else is allowed: approximate or learned feedback pathways, purely local plasticity, fixed (innate) circuits with learning confined to their interfaces, fast associative memories, energy-based inference, evolutionary or other non-gradient outer loops. The requirement is generalization to unseen inputs, not memorization.

Reference. Backpropagation through time on the same network reaches 98-100% on held-out single operations, 72-83% on untaught operator pairs, and 82-85% at gaps three times the trained length when every byte is a prediction target.

Already established in our own experiments (do not re-derive):
- Sign-symmetric feedback through time matches the reference on held-out single operations and loses about 30 points on untaught compositions; Kolen-Pollack learned feedback on the recurrent matrix alone recovers most of that loss.
- With every byte as a target, sign-symmetric feedback fails to extrapolate in gap length, and the failure is tied to error arriving from unpredictable positions.
- A random sparse expansion with a local readout (mushroom-body-like) memorizes and does not generalize, alone or as a residual beside a shared network.
- A ring module with hard-wired modular operations answers untaught compositions at 100% once its input and output interfaces are accurate; those interfaces were learned by backpropagation with the parsing supplied.

Questions.
1. Beyond backpropagation, not merely matching it. Which mechanisms that avoid end-to-end backpropagation have outperformed a tuned backpropagation baseline on the same architecture and data, on any of: (a) held-out or out-of-distribution accuracy, including systematic or compositional generalization and length extrapolation; (b) few-shot or one-shot acquisition; (c) continual learning without replay; (d) online learning from a non-stationary stream? Check at least: predictive coding and "prospective configuration", equilibrium propagation and other energy-based schemes, target propagation, forward-only schemes, Hebbian or competitive unsupervised learning with a local readout, local contrastive or predictive objectives, eligibility-trace rules for recurrent networks, and feedback-alignment variants. For each claimed win give the effect size, the regime in which it holds and the regime in which it disappears. Results on recurrent or sequence tasks, and on held-out data rather than training speed, matter most.
2. Structure in place of gradients. Which systems obtain generalization from fixed or innate computational structure, with learning restricted to interfaces or associations, and how were the interfaces learned without backpropagation? Cover: (a) insect-brain-inspired models: mushroom-body models that go beyond elemental association (patterning, same/different, numerosity, sequences), central-complex ring-attractor and vector-computation models applied outside navigation, and combined models; (b) connectome-constrained or connectome-initialized networks compared with degree-matched rewired controls on learning or generalization, not only on predicting neural activity; (c) vector-symbolic or hyperdimensional architectures, and residue or phasor codes, for modular arithmetic and compositional binding; (d) attractor or grid-code scaffolds used for arithmetic, analogy or relational generalization; (e) built-in arithmetic or algebraic modules, and whether any were trained without backpropagation. For each: what was wired, what was learned, by which rule, on which held-out split, and which control showed that the structure mattered.
3. Fast memory beside a slow learner. Is there evidence that a sparse-expansion associative memory (mushroom-body-like, sparse distributed memory, key-value memory with fixed random keys) added to a learned network gives (a) one-shot binding of new symbols that the network then uses in computation, or (b) continual learning without replay, better than a matched generic index (nearest neighbour over the same states) and better than replay at equal compute? State whether the feature extractor was pretrained by backpropagation and frozen, and what happens when it keeps learning.
4. Non-gradient outer loops. Has any plasticity rule or circuit found by evolution or another outer loop that does not backpropagate through the inner loop produced within-lifetime learning that generalizes to new tasks, or that beats gradient descent in sample efficiency?
5. Theory that bounds the search. What can fixed random features with a trained readout generalize on, and what can they not (lazy or kernel regime versus feature learning), specifically for modular arithmetic and parity-like tasks? Are there sample-complexity results showing an advantage of built-in group structure over learned features?
6. Negative results and failed replications for all of the above, including scaling studies.

Output.
(i) One table, a row per study: mechanism; whether backpropagation is used anywhere, and where; task; comparison; result with numbers; regime; seeds; reproduced or not; URL.
(ii) At most six mechanisms ranked by the strength of evidence for generalization beyond a backpropagation baseline under our constraint, each with the cheapest decisive experiment in a setting like ours and the result that would falsify it.
(iii) Claims you could not verify from a primary source.
(iv) Anything suggesting that the question is ill-posed.
```

##### 二、如何检验"果蝇结构能否作为基础"

###### 1. 要检验的命题

以取自果蝇的固定回路提供共享计算或记忆，学习只在接口上进行，全程不用全局反向传播。这样的系统在同一套泛化测验上不低于"通用网络加反向传播"，并至少在一项预先指定的指标上更高。

| 指标 | 通用网络加反向传播 | 理想上限 |
|---|---|---|
| 未见过的操作数对，按教过的比例计（5%、10%、25%、50%） | 50% 时 98–100%；早期窗口网络实验中一成时只会记忆 | 100% |
| 间隔外推（40 个字母） | 82–85%（全字节），92–99%（仅答案） | 100% |
| 未教过的运算组合 | 72–83% | 100% |
| 新运算族的少样本习得 | 未测 | 待定 |
| 无重放的保持 | 约 1% | 接近习得水平 |

每个候选同时与三者比较：

- **通用网络加反向传播**：通行的基线。
- **同一结构加反向传播**：用来区分结构的贡献和规则的贡献。
- **规模相同但结构被打乱的对照**：确认起作用的是结构本身。

###### 2. 此前约 220 个版本已经回答了什么

- **已回答，不必再试。**
  - 蘑菇体式随机编码加局部可塑性，自己发现不了规则。
  - 把它作为残差并入一个会泛化的共享通路，也没有可靠增益（按种子为 +4.2、+2.5、−4.0）。
- **正面，但依赖给定的结构。** 工程化表示上的加法（COV-R）、给定操作数位置的算子复用（V193）、先天环（E10）。共同点是：结构一旦给定，泛化就出现。
- **没有回答的只剩两件事。**
  - 从字节流到先天回路的接口，能否不靠全局反向传播学出来。
  - 一小组先天回路能覆盖多宽的任务。

新实验只检验这两件事。

###### 3. 设计原则

此前的失败之所以不能定位，记录里有四个原因：学习器、任务和门槛同时改变；任务连参照学习器也学不会；成功依赖脚手架；没有止损。对应五条原则：

1. **可解性在先。** 只用反向传播参照已能泛化的任务。
2. **上界在先。** 先确认接口给定时回路能完成任务，再去学接口。
3. **一套装置，一次只变一件事。**
4. **结构对照。** 打乱结构，规模不变。
5. **预先登记。** 标准、预算上限和止损规则都在运行前写定；失败则记录原因，不开新版本。

###### 4. 方案

**主线：先天计算回路。** 这是三类结构中唯一能直接产生规则泛化的一类。

- **第 0 步，定清单。** 有两种取法。
  - 甲：像 E10 那样按模数把运算接好。它的上界已由 E10 给出，但泛化几乎全由接线保证。
  - 乙：只接与模数无关的原语（角度相加的环、比较器），数制由接口去学。
  - 只有乙能检验"基础"这个命题，我建议取乙。
  - 同时预先指定一个设计时不看的留出任务族。
- **第 0 步的预备测量。** 在已存权重上看会泛化的 GRU 是否自己学出了环状编码。这不需要训练，我可以做，结果用于确定清单。
- **第 1 步，参照。** 允许反向传播，仅作参照。
  - GRU 作控制器加先天环，接口和路由从带干扰字母的字节流里学，不提供解析。
  - 对照为打乱的环和单独的 GRU。
  - 止损：若三颗种子中至少两颗用不上环，说明障碍在接口，本线停止。
- **第 2 步，候选。** 同一结构用界内规则出生。若反向传播能而界内规则不能，做一轮单开关定位，只做一轮。
- **第 3 步，广度。** 在留出任务族上比较达到标准所需的样本数。
- **判定。** 以下三项都满足，命题成立；否则不成立。
  - 第 2 步在低比例的未见操作数对和未教组合上高于通用网络加反向传播。
  - 打乱对照达不到。
  - 第 3 步有少样本优势。
- **预算。** 三个实验，各约 8–10 CPU 小时；新模块需要两份独立实现。

**副线：快速记忆。** 它不产生规则泛化，作用在保持和一次性绑定上。

- **先做零训练的上界。** 在已存网络的内部状态上建稀疏键存储，对照同等容量的通用最近邻索引，测一次写入后的召回、容量和干扰。我可以用现有读数程序完成，不占工作机。
- **止损。** 不优于通用索引则停止；优于，再做一个无重放保持的实验。

**我的主观先验**：主线第 1 步通过约六成，第 2 步约三到四成；副线上界优于通用索引约三成。

###### 5. 需要您决定

- 清单取甲还是乙。
- 留出任务族选什么。
- 预算上限是否接受。

##### 三、对第 3 点的理解

- **界内**：任何不含全局反向传播的机制，包括先天回路、快速记忆和非梯度的外层搜索。
- **两个假设，请您纠正。**
  - 候选在发育阶段也不得使用反向传播，沿用"候选以自身规则出生"的裁定。
  - 固定回路自带的固定反馈通路在界内，因为它不搬运任何学到的权重。
- **验收标准是泛化。** 我建议以上表作为所有候选的统一验收；反向传播只作参照。

## 37. 2026-11-15 18:06 UTC — Mr. Xu

> I have read the old ring module and the current birth harness. Now checking the fixed operations for both inventories numerically before I write them into the briefs.
>
> Both inventories' fixed operations pass exhaustive tests. Writing the two briefs now, with the shared apparatus stated identically in each.
>
> 两份任务书已写好并推送（提交 907c9fc4，已在远端确认），甲、乙各一份。两份都尚未发布，等审稿模型审过再给工作者。
>
> - 甲：`OPUS_TASK_E35A_INNATE_CIRCUITS_INVENTORY_JIA_20261115.md`
> - 乙：`OPUS_TASK_E35B_INNATE_CIRCUITS_INVENTORY_YI_20261115.md`
>
> ## 两份共同的部分
>
> 两份都只做备忘录里的第 1 步，也就是参照阶段：允许反向传播，不产生任何候选规则。
>
> - **装置相同，只换模块。** 控制器是 E26 的 GRU。它用自己的状态算出写入门、运算权重和读出门，模块看不到字节、位置或角色，即不提供解析。
> - **三个组。** GRU 单独、GRU 加固定模块、GRU 加打乱的模块。GRU 单独组与 E26 的反向传播组逐位相同，用已存的哈希核对；这也让两位工作者的训练循环互相校验。
> - **在低教学比例上比较。** 教过的操作数对取 10% 和 25%。50% 时 GRU 单独已达 98%，没有提升空间；低比例下它是否留有空间尚不知道，这是结果表的第 0 行。
> - **先做上界。** 第 0 阶段在提供解析的条件下确认模块能完成任务，不过关就不进入字节流阶段。
> - **止损规则。** 沿用审稿模型的规则：三颗种子中至少两颗用不上模块，这条线就停。"用上模块"定义为去掉模块读出后准确率至少下降 15 个百分点。
>
> ## 两份的差别
>
> | | 甲 | 乙 |
> |---|---|---|
> | 接好的东西 | 模 89 的四种运算，同 E10 | 相位相加，以及按比较结果放行整个编码的比较器 |
> | 要学的东西 | 接口：何时写入、选哪种运算、如何读出 | 接口，加上整个数制（模数也由它学出） |
> | 打乱对照 | 随机运算表 | 随机双线性映射 |
> | 第一阶段预算 | 六次出生，约 7–9 CPU 小时 | 四次出生，约 5–7 CPU 小时 |
> | 我对第 0 阶段过关的估计 | 80% | 45% |
>
> 乙里"与模数无关的原语"具体是什么，是我定的，审稿模型可能有不同想法。
>
> ## 我核对过的
>
> 固定运算在写进任务书前都穷举测过，没有做任何训练：
> - 甲的四种运算在全部 7,921 对上精确。
> - 随机表只有 1.4% 的对与加法一致。
> - 乙的原语在精确相位编码下能解出全部四种运算，说明精确的接口存在；学习能否找到它正是要检验的。
>
> ## 没有在这两份里决定的
>
> - **留出任务族：** 只有第 1 步通过后才需要，届时我提出方案。
> - **副线和预备测量：** 快速记忆的零训练上界，以及 GRU 是否已学出环状编码，审稿模型说可以自己做，不占工作者。
> - **文献检索：** 提示词在备忘录里，由您发起。
>
> ## 给审稿模型的消息
>
> "请审 907c9fc4 上的两份 E35 任务书。检查要点：
> 1. 共用装置（寄存器、写入门、运算权重、读出门）是否确实不提供解析，是否可学？
> 2. 乙的两个原语是否符合你说的'与模数无关'，比较器按标量放行整个编码是否合适？
> 3. 两种打乱对照是否真的去掉了结构，而且接口无法绕过？
> 4. 用 10% 和 25% 的教学比例作为比较点是否合适？
> 5. 标准、按比例读取的结果表和止损规则是否完整、可计算？
> 6. 第 0 阶段的门槛（25% 比例下三颗种子中两颗达到 90）是否合理？
> 7. 模块固定运算以穷举测试代替第二份独立实现，是否可以接受？"
>
> 之前的两条工作者消息（E33 第二阶段和 E34）仍然有效；两个工作分支目前还没有动静。

## 38. 2026-11-15 18:12 UTC — Reviewer

#### 两份 E35 任务书的审查（提交 `907c9fc4`）：设计可行，有四处必须修改

两份任务书都忠实于我提出的方案，固定运算我独立复核无误。但止损规则会把两种不同的失败混为一谈，上界阶段的训练量也有风险。四处必改都是文字修改。

复核脚本在 `claude/vibrant-cerf-sje726` 的 `6858f25c`：

```
D=origin/codex/minifly-consolidated-20261023; M=MOD97_LOCAL_RULE_20261106; R=PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109
python3 -I $R/e35_brief_checks.py
```

##### 1. 共用装置：是否不提供解析，是否可学

不提供解析这一点成立：前向只接收字节，所有门都由 GRU 自身状态算出。原则上可学，但有两个问题。

- **止损规则混淆了两种失败（必改）。** GRU 的直接通路可以把教过的题背下来，训练损失随之消失，模块通路就得不到梯度。这样"没用上模块"可能是因为背诵更快，而不是接口学不会。现在的止损规则会把前者也读成"障碍在接口"并关闭整条线。
  - 修改：出现第 3 行时，停止之前先跑一个强制组。把 `l3` 输出中十个数字字节的逻辑值屏蔽掉，数字只能经模块读出。这是按类别屏蔽，不是按位置，不算提供解析。
  - 强制组能泛化而 `M` 不能，障碍是竞争；强制组也不能，才是接口，止损规则才适用。
- **接线本身没有功能性检验（必改）。** 检查 3 和 4 只验证模块不影响控制器、消融精确，没有验证寄存器、运算、读出这条通路接对了。接线若有错，结果同样是第 3 行。
  - 修改：增加一项不需训练的检验。用人工构造的状态和手工设定的接口，确认整条通路在全部 7,921 对、四种运算、0、12、40 个字母下给出正确结果。乙可只检查结果编码，不检查读出层。

##### 2. 乙的原语

符合我说的"与模数无关"，比较器按标量放行整个编码也合适。

- **模块中没有任何与 89 有关的量**，精确相位编码下四种运算全部正确，我的复核与协调者一致。
- **泛化确实不由接线保证。** 精确编码下读出只需区分 89 个结果编码；编码若不是同态（随机角度），需要区分 4,005 个。学习能否找到同态，就是这个实验要问的问题。
- **建议一：加一个更容易优化的写法。** 角度直接参数化容易卡在局部解上。上界阶段可加一个复数参数化（直接输出二维向量，不经角度），只多几分钟。
- **建议二：加一个最易接口的对照。** 上界阶段从数字到编码经过一层 tanh，流式阶段是 GRU 状态的线性读出，两者不同，所以上界阶段并不严格是上界。可加一个"每位数字各查一张角度表再相加"的变体：它若也不泛化，停止的理由才充分。

##### 3. 打乱对照

两种对照都去掉了结构，接口绕不过去。

- **甲。** 随机表与加法只在 1.2–1.4% 的对上一致；356 列中没有一列对另一个操作数是一一对应的，平均每列只有 56 个不同输出。接口无法靠重新标号把它变回加法。
- **乙。** 随机双线性映射不含循环群结构。这一条是我推理的结论，没有做数值验证。
- **一个不对称，建议写进"不能说明什么"。** 固定模块可以充当"直通管道"：甲中 max(A, 0) 原样返回 A，乙中寄存器 B 不写入时相乘原样返回 A。控制器可以自己背下答案再经模块送出，这样"使用了模块"的标准会被满足而结构并未被使用。第 1 行另有"高于打乱对照 15 分"的要求，所以读法不受影响。

##### 4. 10% 和 25% 两个教学比例

合适，另有一项建议。

- **覆盖没有问题。** 两个比例下，89 个数值在两个操作数位置上都出现过。
- **10% 时覆盖很薄。** 个别数值在某一任务的某个位置只出现 1–2 次（25% 时至少 10 次），所以 10% 的结果同时检验了数码在任务之间的迁移。
- **建议先跑 `G`。** GRU 单独是否留有空间尚不知道。先跑它的两次出生，模块组只在有空间的比例上跑，最多可省四次出生。

##### 5. 标准、结果表、止损规则

可计算，基本完整。

- **第 3 行的条件有一处不妥（必改）。** 若 `M` 高出 `G` 和 `X` 各 15 分以上，但消融后下降不足 15 分，按现在的写法落入第 3 行，被读成"没有学会使用回路"。这种情况应归入第 4 行，不作解读。
- **建议按运算族分别读。** 加减与比较大小对 GRU 的难度差别很大。四种运算取平均后，可能出现加减有空间而平均值显示"无空间"的情况。空间和各行应对加减、max/min 分别判定。

##### 6. 第 0 阶段的门槛

90 分的门槛本身合理，问题在训练量。

- **甲的训练量未经检验（必改）。**
  - E10 在 8,192 次更新时没有收敛，记录里是 32,768 次才成立。
  - 即使那样，一颗种子的单步准确率也只有 62–93%。
  - 任务书用的是 16,384 次，比例还更低。门槛可能因训练量不足而失败，并错误地终止后续阶段。
  - `git show $D:$M/exp10_ring.py | sed -n '33p;175p'`；`git show $D:$M/E45_REPORT.md | sed -n '207,209p'`
  - 修改：用 32,768 次，并每 4,096 次记录一次留出准确率。
- **乙建议同样加长并记录曲线，** 同时加跑流式阶段的优化器设置（学习率 0.001，衰减 0.25）。同态解的形成通常较晚，也依赖权重衰减；上界阶段只需几分钟，不应因设置不同而过早否定。

##### 7. 以穷举测试代替第二份实现

对固定运算本身可以接受，而且比第二份实现更强；我用 numpy 按任务书公式重写了一遍，全部一致。它不覆盖接线，接线由第 1 点的功能性检验补上。

另外两处事实核对无误：

- 干出生参照存在，哈希前缀为 `adf933d7`。
- `make_opt` 就是 `adamw16(m, LR, WD, ())` 加一条断言，所以模块组的优化器与 `G` 一致。

##### 结论

现在还不能发给工作者。必改四项：

1. 止损之前先跑强制组，并改写止损规则的解读。
2. 增加理想接口下的整条通路检验。
3. 第 3 行条件改为只含"`M` 与 `G` 相差不足 15 分"；"高于 `G` 但消融不降"归第 4 行。
4. 甲的第 0 阶段用 32,768 次更新并记录曲线。

建议五项：先跑 `G`；按运算族分别读；乙的第 0 阶段加长、加跑流式优化器；乙加复数参数化和按位查表两个变体；把直通管道写进"不能说明什么"。

四项必改照此完成后，我不需要再审。

##### 没有检查的

- 我没有运行任何训练。
- 乙的打乱对照无法被绕过，是推理而非数值结论。
- 直接通路与模块通路的竞争是我的推断；它会不会发生，正是强制组要回答的。

## 39. 2026-11-15 18:27 UTC — Mr. Xu

> (an uploaded literature report) Discuss next steps given the research report. Note our goal is persistent learning, not conventional training+validation+test data set setting. So the main problem is given fixed memory / computation capacity, how does local learning beat naive back-propagation in generalization tasks.

## 40. 2026-11-15 18:36 UTC — Reviewer

**Position.** Under a fixed memory and compute budget, a local rule will not beat backpropagation by assigning credit better. The report, the cost arithmetic and our own record agree on that. The one axis where an advantage is plausible and unmeasured is the experience needed to generalise: fixed circuits with a small plastic interface should need far fewer examples of a new task than a generic network. The next steps should measure that quantity in a single-pass stream and stop measuring parity.

##### 1. How the persistent-learning goal changes the reading of the report

The report's headline is that local rules win only where backpropagation is "handicapped". Its handicaps are a single pass, small batches, a truncated window and no revisiting, which are the conditions of persistent learning. That does not make the report favourable. The studies in that regime are the weakest it cites: effects shown only in figures, 32-unit networks, untuned baselines. The literature there is thin, not positive.

##### 2. What our own record already says

| Possible source of advantage | Evidence | Verdict |
|---|---|---|
| Better credit assignment | Sweep and backprop births extrapolate equally with answer targets (40 letters: 96.0 / 97.0 / 99.7 against 97.7 / 98.7 / 92.3). Kolen–Pollack feedback matches exact recurrent feedback on two of three seeds, and neither exceeds the backprop life (E31). | Parity at best |
| Lower cost | The sweep stores and computes what backprop through time does, and Kolen–Pollack adds a matrix. Per-synapse traces save memory only when dependencies exceed roughly 400 bytes (my arithmetic); ours are at most 50. On the GRU they failed (9.10: 2–14% against 68–86%). | None at this scale |
| Retention | Without replay, both rules lose everything when the stream changes (9.88: pairs learned to 61–92, ending near 1). A bounded store of raw past bytes, 0.26–1 MB, restores 0.81–0.98 of the joint-training reference under the local rule (9.42, 9.46). | Solved cheaply under either rule |
| Experience needed to generalise | With the operations innate and parsing supplied, untaught compositions were answered without teaching (E10: 100 / 65 / 91). The report finds the only large out-of-range effects come from fixed group codes. | Open |

##### 3. Where an advantage can come from

A fixed circuit changes what has to be learned. A generic learner must discover the group structure of addition mod 89. The theory the report cites puts that at the order of p^(5/3) examples for a feature-learning network, and a constant fraction of all p² pairs for any fixed-feature learner. The second figure also explains why the mushroom-body learner memorised. With the operation wired in, what remains is a map from about twenty digit-position symbols to codes, plus a choice of operator per task code.

In a stream this predicts three measurable things:

- **Acquisition:** fewer presentations before never-seen pairs are answered.
- **Replay:** less of it is needed, because the operators cannot be overwritten. Replay is paid for in acquisition (E9: 65–70% without it against 44–50% with it).
- **Composition:** untaught combinations arrive at once.

All three would also hold under backprop on the same architecture. The claim available to us is therefore: fixed circuits with a locally learned interface beat a generic network under backprop at equal memory and compute. Two controls say what the gain is owed to: the same circuit under backprop, and a scrambled circuit under the same rule. Nobody has learned such an interface without backprop; that is the experiment.

##### 4. Next steps

1. **Fix the claim and its measures before more runs** (half a page, no compute).
   - Budget: every stored number (weights, optimiser state, feedback copies, the credit window, the replay store) and operations per byte, equal across arms.
   - Measures, all taken on the stream itself, so no train/validation/test split is needed:
     - accuracy at first encounter on never-seen pairs, each item answered before its error is applied;
     - presentations to criterion for each new family;
     - retention probes on earlier families.
2. **E35 continues as released** (revision 2 at `dc05107d` adopted all four required changes). Add taught fractions 0.03 and 0.05 once the GRU-alone arm shows room. The separation should be largest where data are scarce; the brief now stops at 0.10. This is about six births, roughly 5 CPU-hours on one seed.
3. **A stream experiment after E35's first stage** (new brief), run only if the module arm beats both the GRU alone and the scrambled control for some family.
   - Families are introduced in sequence, single pass.
   - Arms: the generic GRU under backprop with and without the replay store, the module arm, and the scrambled arm.
   - If E35 instead ends at its stop rule, the structural line closes. On present evidence the defensible result is then parity under locality, not an advantage.
4. **E32–E34 are enabling work.** A life without an answer mask needs every-byte targets, so E34 should finish, but the line should not be extended. Add one backprop birth with the same entropy gain only if a sweep arm exceeds the backprop reference of 82.0 at 40 letters.
5. **Deferred:**
   - Prospective configuration needs a new learner and has no sequence evidence.
   - Evolutionary search cannot cover an interface that includes a parser inside about a million weights. Keep it for the few selection parameters if the forced arm fails.
   - Fast memory stores and does not generalise, by both the record and the report. I can run the nearest-neighbour control on saved states without a worker if you want it.

##### Decision for you

Should the comparator in step 1 include the replay store? I recommend reporting both and stating the claim against the one with it. The store costs less memory than the weights, and a result that holds only without it would be a result about replay.

##### Limits

- The report's quotations came through a text extractor, its figures were not inspected, and several sources are preprints weeks old. I opened none of them.
- I rely on the report only where it agrees with our receipts, plus the kernel bound, whose constants are unspecified.
- The 400-byte figure is my own approximate arithmetic.
- No new results from E33 stage 2, E34 or E35 are on the default branch yet.

## 41. 2026-11-15 18:40 UTC — Mr. Xu

> I repeatedly said as long as we do not do full backpropagation, anything is okay. Why keep asking me? I do not understand the point. But my question to you - do you need an updated deep research session, or we are okay now?

## 42. 2026-11-15 18:41 UTC — Reviewer

You are right, and I will stop asking. The rule is settled: anything short of full backpropagation is admissible. That covers the sweep, Kolen–Pollack feedback, gradients confined to a block, evolutionary search and delta-rule readouts.

The question in my last message was about the baseline, not the rule, but it was mine to decide. I will report backpropagation both with and without the replay store and state any claim against the version with it.

**On deep research: we are okay now.** Nothing that is running or about to be briefed depends on more literature. E35's first stages use backpropagation as the reference, and the stream experiment's baselines are already in our own record.

I do recommend one narrow follow-up, to run in parallel rather than as a redo. The report's central negative is that nobody has learned the interface into a fixed circuit without backpropagation. By its own account that rests on incomplete searching: it ran one query on vector-symbolic interfaces and did not open the central-complex work. That negative decides how E35's later candidate stage is designed, so the answer is worth having before E35 reads out. The report also did not cover online learning under matched memory and compute budgets, which is your framing.

The prompt below is ready to paste and has no project details. The named papers are leads from my memory, not checked here.

```
Targeted follow-up literature search. Two questions only. For each source,
open the primary paper and state: the learning rule exactly, whether any
backpropagation is used anywhere, the task, the comparison, effect sizes and
seeds. Mark anything read only as an abstract.

Q1. Has a map from raw inputs into a fixed structured circuit (ring
attractor, grid or residue code, vector-symbolic codebook, hard-wired
arithmetic unit) been learned by a rule with no end-to-end backpropagation
(Hebbian, reward-modulated, local predictive error, evolutionary) and then
tested on inputs or combinations not seen in training? Start from:
plasticity of visual inputs onto the fly heading ring (Fisher et al. 2019;
Kim et al. 2019), local-rule learning of path integration in ring attractors
(Vafidis et al. 2022), Hebbian learning of sensory-to-grid mappings, codebook
learning for resonator networks, and anything since 2023. Report failures as
well as successes. For each, give the number of learned interface parameters
and say whether the system had to decide which inputs are relevant (a parse
or gate).

Q2. In single-pass, non-stationary streams evaluated online (each input
scored before it is learned from) under an explicit, equal budget of memory
and computation:
(a) which methods beat plain online gradient descent with a small replay
    buffer once budgets are matched, and by how much (start from work on
    computationally budgeted and real-time online continual learning, 2022
    onward);
(b) does a modular or fixed-structure architecture reduce the examples
    needed for a new task, or the replay needed to keep old ones, against a
    monolithic network of equal size (modular and compositional continual
    learning; group-structured networks; sample-complexity theorems for
    invariant or equivariant learners, e.g. Bietti et al. 2021, Elesedy and
    Zaidi 2021);
(c) any such result on sequence or algorithmic tasks rather than image
    classification.

Output: one table per question, then one paragraph each on what is
established, what is contested, and what was not found, with the searches
tried.
```

## 43. 2026-11-15 18:54 UTC — Mr. Xu

> (an uploaded literature report)

## 44. 2026-11-15 18:57 UTC — Reviewer

The second report closes the literature question: no further research session is needed, and nothing now running changes. It adds three things to the briefs that come after E35's first stages.

##### What it settles

- **A local rule can learn the map into a fixed circuit, under one condition.** In every case found (fly compass, grid models), the circuit's own state was driven by a separate trusted channel and served as the teacher. Tests stayed inside the trained range.
- **Our experiment is not in the literature.** No source tests such an interface on untaught combinations, and none has the system decide which inputs are operands. Both reports end at "not found" in the same place.
- **Plain replay is the baseline to beat under matched budgets.** No budget-matched comparison exists for sequence streams or for any learner without backpropagation, so the stream experiment would be new.

##### What it changes

1. **E35's candidate stage (not yet briefed) gets two candidates, in this order.**
   - First, the sweep through the fixed module, which uses existing code.
   - Second, a teacher-driven rule: fix the read-out from ring state to answer bytes, so each taught answer sets the ring's target state, and learn the byte-to-code map by a delta rule against it.
   - Fixing one side also removes an ambiguity. When both sides are learned, many codes are equally consistent with addition, and the compass papers report that as several stable offsets. This is my inference and the report's, and it is untested.
2. **The stream experiment's scorer gets two controls.**
   - A blind predictor that answers from recent history without reading the operands, reported beside every online score. On never-seen pairs it should score about 1–2%, by my arithmetic.
   - A delayed score for each family after the stream has left it.
3. **The low taught fractions I proposed need a split score.** The report's clearest failure is that states never visited in training stay unmapped, so I counted how many of the 89 operand values never appear in a given position within a task:

   | Taught fraction | Values missing per task and position | Missing in all four tasks |
   |---|---|---|
   | 0.03 | 1–10 | 0 |
   | 0.05 | 0–2 | 0 |
   | 0.10 | 0 | 0 |

   At 0.03 and 0.05, held-out pairs must therefore be scored in two groups: both operands seen in that position in that task, or not. Otherwise unseen pairs and transfer of a number's code across tasks are mixed.

   ```
   python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e35_low_fraction_coverage.py
   ```
   The script is on `claude/vibrant-cerf-sje726` at `e6109076`.

##### Limits

- The report could not read the equations or supplements of its key sources, and I opened none of them.
- I ran no training; the coverage count is the only new measurement.

## 45. 2026-11-15 18:58 UTC — Mr. Xu

> So now given the circuit structure itself is shaped by learning in our setting, the literature is only of limited help. Our neural network structure would change with time. Discuss next steps.

## 46. 2026-11-15 19:25 UTC — Reviewer

**Your premise holds, and the saved networks show it.** I read the six births and nine lives we already have (three seeds; sweep, Kolen–Pollack and backprop), without training anything. The learner builds its own answer circuit, a life rebuilds the weights that carry it, and what gets lost is specific and small. Both rules behave alike, so the next experiments should be about what the learner keeps, not about the rule or a hand-wired circuit.

##### What the saved networks show

| Finding | Evidence |
|---|---|
| **The network makes its own ring.** At `=`, every trained birth holds a compact circular code of the answer to add and sub mod 89. | 85–95% of the code lies in two components, one for the size of the answer and one for its units digit, in all six births. An untrained network has none. |
| **A life rebuilds the weights but keeps the form of the code.** | Over a life the recurrent weights move by 1.12–1.17 times their size at birth (seed 73129). The old and the new families still show the same two-component code in all nine lives. |
| **What a life loses is the old family's own parameter.** The life never rehearses the mod-89 tasks. | Pairs that need the wrap at 89 are answered right 0–2% of the time in all nine lives (92–100% at birth); pairs that need no wrap, 10–93%. In two backprop lives the old subtraction prompts are answered as subtraction mod 97 (88% and 91%). |
| **Later families are learned quickly from fewer examples.** | The third life family (a quarter of pairs taught) reaches 77–94% on held-out pairs after 131,000 items. The birth (half of pairs taught) was at 63–84% after 262,000. |

Putting one group of weights (input, recurrent or read-out) back to its birth value does not restore the old skill, so the loss is not confined to one layer.

##### What this changes

- **The thing to study is the structure the learner builds and what time does to it.** It can be read directly from the weights, and the scripts for that are now pushed.
- **The rule is again not the difference.** Sweep, Kolen–Pollack and backprop build the same code and lose the same thing.
- **A likely cause of the loss is in our own settings.** Weight decay 0.25 at learning rate 0.001 shrinks any weight the current data do not support to 21% within one life block. This is my arithmetic, not a measurement, and it applies to both rules.

##### Next steps

1. **Reminder test (small worker job).** From each saved life end, teach the mod-89 family again and score every 64 updates, split by wrap. If the skill returns within a few hundred updates, retention is a matter of reminding, and its price can be stated in updates. The birth needed 4,096 updates to reach 63–84%. My estimate is about half a CPU-hour for the nine lives.
2. **What erases the old family (about 3 CPU-hours by my estimate).** Six two-block lives with the birth family unrehearsed, under the sweep and under backprop, with three decay settings:
   - decay as now;
   - decay off during the life;
   - decay pulling each weight toward a slow copy of itself instead of toward zero.

   The third is the protection step the record planned and never ran (9.82, 9.74). If it keeps the old modulus without replay and still learns the new family, that is the first thing our learner does that naive backprop does not, at equal compute. The same device could be added to backprop, so the claim would be against naive backprop only.
3. **E35 continues through its first stages only.** Its GRU-alone arm at a quarter of pairs is the from-scratch reference that the fourth finding lacks. I withdraw the later stage I proposed last time (a local rule into a hand-wired ring); it answers a question we no longer have.
4. **No further literature session.**

##### Limits

- The lives were not designed for this; the readings are one pass over saved files.
- The code readings describe the form of the code, not whether it holds the right answer; the wrap split covers that.
- A linear read-out of the answer was too weak an instrument, so nothing rests on it.

```
R=PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109   # claude/vibrant-cerf-sje726 at 784e64b7
python3 -I $R/learned_structure.py add89,sub89,add97,sub97 e26/e26_dev_S_73129.pt e26/e26_life_S_73129.pt
python3 -I $R/life_answers_on_birth_tasks.py e26/e26_life_S_73129.pt e26/e26_life_B_73129.pt e31/e31_life_K_73129.pt
python3 -I $R/graft_birth_into_life.py add89,sub89,add97,sub97 e26/e26_dev_S_73129.pt e26/e26_life_S_73129.pt
python3 -I $R/acquisition_from_receipts.py
```

## 47. 2026-11-15 19:42 UTC — Mr. Xu

> Write an appropriate memo for the coordinator to implement.

## 48. 2026-11-15 19:51 UTC — Reviewer

The memo is written and pushed to `claude/vibrant-cerf-sje726` at `4fd03ca8`:

`PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/MEMO_TO_COORDINATOR_LEARNED_STRUCTURE_20261115.md`

It asks the coordinator for four things:

1. **Brief three small jobs, each audited by me before a worker starts.**

   | Job | Question | Size (my estimate) |
   |---|---|---|
   | A, reminder test | After a life, how many updates bring the unrehearsed birth family back, against learning it from nothing? | 15 short runs, 0.6 CPU-hours |
   | C, fresh reference | Does a learner that has lived need fewer items for a new family than a fresh one? | 6 short births, 0.9 CPU-hours |
   | B, what erases it | Is the old family lost to weight decay or to the new family's updates, and does pulling each weight toward its birth value protect it? | 10 two-block lives on one seed, 4.7 CPU-hours |

2. **Let E35 run through its released stages and brief nothing after them.**
3. **Score every earlier family at every stage of a life from now on**, with add and sub split by whether the pair needs the wrap.
4. **Record your three statements of today**, quoted verbatim, and the fact that the E26 and E31 lives never rehearsed the birth family and lost it.

The memo also carries the findings with the command for each, proposed marks and reading tables for the three jobs, and my expectations on record before any run.

Two things differ from what I told you earlier:

- **Job B uses a fixed anchor at the birth weights, not a slow copy.** Over two blocks the two are the same, and the anchor has no time constant to choose. I added a second strength and a rehearsal arm as the reference. The slow copy comes back only if the screen passes.
- **Job C is new.** Without it, the observation that later families are learned from fewer examples cannot be called a saving.

One figure is stated more precisely than in my last message. Of the pairs that need the wrap, two backprop lives answer 89% and 83% of the old subtraction prompts as subtraction mod 97. The earlier 88% and 91% were shares of all pairs.

A message you can relay: "Read `MEMO_TO_COORDINATOR_LEARNED_STRUCTURE_20261115.md` on `claude/vibrant-cerf-sje726` at `4fd03ca8`. Section 1 lists what is asked. Send each brief back for audit before any worker starts."

## 49. 2026-11-15 19:53 UTC — Mr. Xu

> We want to avoid back and forth, E33 and E34 has shipped. See if this changes anything.

## 50. 2026-11-15 19:58 UTC — Reviewer

E33 and E34 change nothing in jobs A, B and C. They add two small items to E34's next stage and one caution about using its result in a life. I folded all of it into the memo as revision 2 (`a8b5ace2`), so the coordinator has one document.

**Check.** Both reports are on the worker's branch, not yet on the default branch. I recomputed every endpoint of the seven new births from the committed weights, and all agree with the reports.

**What they settle**

- **E33:** on three seeds, a sweep birth with distractor bytes as targets does not extrapolate to 40 letters (24, 29, 32), and one without them does (94, 95, 87). That question is answered.
- **E34:** on one seed, scaling each byte's error by the network's own confidence, at the sharpest setting tried, learns in range and extrapolates with every byte a target and no answer mask. It is a candidate on one seed, nothing more.

**What changes**

1. **E34's next stage runs as briefed, plus one backprop birth with the same confidence scaling** (about 0.8 CPU-hours, my estimate). The sweep with scaling now ends near plain backprop at 40 letters: 79.8 against 82.0 on the registered measure, 99.3 against 84.7 on the final row. If the two further seeds put it above, that is a second possible claim against naive backprop, at no extra memory. The added birth says whether the gain belongs to the rule or to the objective.
2. **Its report should show when each arm learns in range, not only where it ends.** The passing setting learns slowly. Accuracy at 12 letters, seed 73129:

   | Updates | 4,096 | 8,192 | 12,288 | 16,384 |
   |---|---|---|---|---|
   | Sweep with scaling (passing setting) | 2 | 6 | 17 | 78 |
   | Plain backprop | 74 | 89 | 95 | 91 |

3. **The scaling should not go into a life yet.** It gives almost no weight to answers the network cannot yet predict, so it holds back exactly what is new. In a life every new family would start that way, and items needed for a new family are the measure we care about.
4. **The registered measure hides a swing.** The passing arm's last three rows at 40 letters are 78, 62 and 99. The next report should give the rows beside the mean.

I did not check rows before 16,384 (the weights are not committed) or the worker's internal checks.

Message to relay: "The memo is now revision 2 at `a8b5ace2`. Section 14 covers E33 and E34; item 5 of section 1 is the only new request."

## 51. 2026-11-15 20:01 UTC — Mr. Xu

> 现在项目处于什么状况？

## 52. 2026-11-15 20:01 UTC — Reviewer

项目现在处在一个换方向的节点上：两条旧线各出了一个结果，新方向的备忘录已经写好，但协调者还没有动作，默认分支从 15 日 13:16 起没有新提交。

##### 已有结论

- **规则本身不是差别所在。** 在回答字节为目标的条件下，sweep 和反向传播学出同样的内部答案编码，丢掉同样的东西，学新任务族也一样快。
- **E33 已完成（三颗种子）。** 让 sweep 无法外推到 40 个字母的是干扰字母作为预测目标，其他非答案字节不影响。
- **E34 第一阶段通过（一颗种子）。** 按网络自身置信度缩放每个字节的误差，在最尖锐的设置下，不用答案掩码也能学会并外推。代价是学得慢，约 16,000 次更新才学会，普通反向传播 8,192 次时已到 89%。
- **学习者自己造出了环形编码。** 六个出生网络在加减法上都有同一种两成分的环形答案编码，未训练网络没有。
- **一生之后丢的是旧任务族的模数。** 九个生命里，出生时学的 mod 89 任务从未被复习。需要按 89 回绕的题，答对率是 0–2%（出生时 92–100%）；不需要回绕的题还能答对 10–93%。

##### 正在跑

- **E35-B（无模数原语）：** 模块组和打乱组在 10% 教学比例上已跑完，结果表在等 E35-A 的 GRU 单独组作参照。
- **E35-A（mod 89 环）：** GRU 单独组跑到 12,288 次更新，还没结束。

##### 等协调者

备忘录第 2 版在 `claude/vibrant-cerf-sje726` 的 `a8b5ace2`，要它做五件事：

1. 为三个小任务写任务书，每份先给我审：
   - A：一生之后，重新教多少次更新能把出生任务族找回来。
   - C：活过一生的学习者学新任务族，是否比全新网络需要更少的题。
   - B：没被复习的任务族是被权重衰减抹掉的，还是被新任务的更新抹掉的；把权重拉向出生值能否保住它。
2. E35 跑完已发布的阶段后不再续写。
3. 以后每份含生命的报告，都对每个较早的任务族逐阶段评分。
4. 把您今天的三条表述，以及"E26/E31 的生命从未复习出生任务族并丢掉了它"这一事实记入记录。
5. 放行 E34 第二阶段（另外两颗种子），外加一次带同样缩放的反向传播出生作对照。

另外，E33 和 E34 的报告还在工作者分支上，没有合并到默认分支。

##### 您的问题目前的答案

"局部学习如何胜过朴素反向传播"还没有成立的结论，有两个候选：

- **不复习也能保住旧任务族**（任务 B）。朴素反向传播在这一点上保不住。
- **每个字节都作目标时的外推**（E34）。在一颗种子上，sweep 加置信度缩放与朴素反向传播接近：登记指标 79.8 对 82.0，最后一行 99.3 对 84.7。要等另外两颗种子。

## 53. 2026-11-15 20:08 UTC — Mr. Xu

> 从哲学角度上来说，最朴素的反向传播只对每个样本进行同样的学习，通过误差进行逐级修正，网络结构是恒定的，虽然可能在RL或其他阶段只是激发网络的一部分。meta learning可以学习最优的某个网络结构，但一般也认为训练的objective是确定的才能做到。但在持续学习情况下，样本之间存在内在结构，学习本身会帮助调整网络结构，不断去自动优化学习样本之间的关系。作为代价静态的backpropagation无法大规模使用。因此我们回到一个全局元学习层次的优化问题。这个问题听起来颇为直观，但为何做了多轮deep search，都无法找到相关信息，而实验本身推进的也颇为缓慢？

## 54. 2026-11-15 20:12 UTC — Reviewer

您的表述在文献里有直接对应，检索没找到，主要是因为我把检索问题问错了层次。实验慢，是因为两周来多数实验在回答另一个问题，而且元层次的搜索一直靠人手逐个假设去做。"没人做过"只成立一半：提法有人提过，缺的是结果。

##### 一、为什么检索不到

**1. 我问错了。** 两轮提示词问的是"哪种局部规则能胜过反向传播"和"固定回路的接口能否局部学出"，都在规则和结构的层次，而且只收有对照、有效应量的结果。您的问题在元层次：学习方式本身随经历改变。提示词把这类文献筛掉了。

**2. 这个问题没有统一的名字。** 我刚核对了四篇，与您的表述逐条对应，项目记录里一篇都没引用过：

- [Kumar 等](https://arxiv.org/abs/2307.04345v3)：把持续学习定义为在计算约束下最大化长期平均回报，也就是您说的"给定固定内存和计算"。
- [Javed 与 Sutton 的"大世界假说"](https://arxiv.org/pdf/2512.23419v1)（经后续论文和摘要转述）：世界永远比学习者大，学习者学不完，只能不停跟踪。
- [Abel 等](https://proceedings.neurips.cc/paper_files/paper/2023/hash/9d8cf1247786d6dfeefeeb53b8b5f6d7-Abstract.html)：最优的学习者永不停止学习，其行为是一个永不结束的隐式搜索。
- [Alberta 计划](https://ar5iv.labs.arxiv.org/html/2208.11173)：以持续学习和元学习为核心，具体机制是给每个权重学一个自己的步长。

我只核对了这四篇存在及其摘要，没有读全文。

**3. 这些是定义和纲领，不是结果。** 两份报告的"未找到"在这一点上是准确的：没有人展示过一个不用完整反向传播、在一生中自我重组、并在同等预算下胜过静态训练的学习者。

**4. 缺结果有两个原因。**

- **主流绕开了它。** 用静态反向传播在海量数据上优化一个固定目标（预测下一个符号），网络的前向计算自己就成了一个学习算法。在持续学习的设定里您说静态反向传播用不了是对的，但主流恰恰靠大规模使用它避开了这个设定，把元层次的问题搬进了数据和上下文窗口。代价是窗口有限，部署后权重不变。我们的约束排除了这条捷径，所以我们站在人少的地方。
- **元层次的全局优化难以评估。** 评估一种学习方式需要一整个生命；没有固定目标就没有公认的基准；对样本流不作任何假设时，没有哪个学习者平均更好。所以总要在某一层固定一些东西。我们固定的是流的生成方式、逐字节的预测误差和预算。

##### 二、对您表述的一点修正

朴素反向传播不变的不是网络结构，而是学习方式。今天的测量里，反向传播的一生中循环权重的变化超过出生时自身的大小（1.12 倍，一颗种子），内部回路照样被重建。不变的是每个样本、每个权重、任何时刻都用同一种更新。

所以在我们这里，元层次要决定的是三件事：

- 哪些样本教得多（E34）；
- 哪些权重可以动（任务 B）；
- 什么时候复习（任务 A）。

E34 是第一个证据：换规则没有差别，而让每个样本教多少取决于网络自身的状态，40 个字母上的准确率从 2% 变成 80%（一颗种子）。

##### 三、为什么实验慢

1. **目标错位。** 多数实验问的是局部规则能否追平反向传播，最好的结果也只是持平。直接研究一生中如何学的只有 E4、E8、E9 和 E16 系列，11 月 10 日起搁置（9.76）。
2. **外层循环是人。** 一个假设、一份任务书、几轮审查、一天的出生，每个实验只回答一个是非题。审查换来了可信度，付出的是周期，这里有我的一份。
3. **噪声大于效应。** 同一次出生的最后三个存档点可以是 78、62、99，只有三颗种子，很多实验以"混合"告终（E32 的十二条预测成立四条）。
4. **先训练，后观察。** 关于原因的假设一直靠训练新网络来检验，多数落空（E27 五条预测全部落空或作废）。今天直接读已保存的网络，就看到丢的是模数。旧任务族成绩下跌的那一列数字从 12 日起就在回执里，没有人看。这一条我有责任：我审的是被要求审的，没有去看网络学出了什么。

##### 四、"没人做过是好消息"

位置确实是空的，但它空着是因为难，而且主流不需要它。所以这说明风险高，不说明容易。新颖性要由结果确立，眼下最接近的两个候选是任务 B 和 E34 第二阶段。

##### 五、需要您决定的一件事

我建议把外层循环变快：用小模数、小网络、短生命，把一个生命压到几分钟（我的估计，未验证），在上面三个控制量上系统扫描，只在完整协议上做确认。这会改变投入方向。您同意的话，我写进备忘录第 3 版。

如果还要一轮检索，下面的提示词对准了这个层次，不含项目信息。我预期找到的是评价方式和三个控制量各自的候选机制，不是现成答案。

```
Literature search on learners whose learning procedure changes with their
own history, in a single non-stationary stream under fixed memory and
compute. For each source: open the primary paper; state the mechanism, where
any end-to-end backpropagation is used, the stream, the budget accounting,
the comparison and effect sizes, seeds. Mark anything read only as abstract.

Q1. Frameworks and evaluation. Formal definitions of continual learning as
bounded-resource learning (computationally constrained RL; the big world
hypothesis; continual RL as never-ending search; the Alberta Plan). What
evaluation protocol does each prescribe, and has any been instantiated on
sequence or algorithmic prediction?

Q2. Mechanisms, in three groups. For each, evidence on non-stationary
streams against plain SGD with a small replay buffer at equal budget, and
any use without end-to-end backpropagation.
(a) Which weights may change: per-weight step-size adaptation (IDBD,
    Autostep and successors), utility-based protection or perturbation,
    continual backprop, metaplastic or multi-timescale synapses.
(b) Which samples teach: weighting by learning progress or prediction gain,
    self-paced and automated curricula, confidence or entropy weighting,
    prioritised replay.
(c) When to rehearse: replay scheduling, consolidation gated by
    predictability, sleep phases.

Q3. Adapting the learning procedure within one lifetime: online
meta-gradients, bootstrapped meta-learning, self-referential or
self-modifying learners, the success-story algorithm. Cost, results against
a tuned fixed learner, and failure modes such as short-horizon bias.

Q4. Negative results, and stated reasons why this area has frameworks but
few controlled wins.

Output: one table per question, then what is established, what is contested,
and what was not found, with the searches tried.
```

Sources:
- [Continual Learning as Computationally Constrained Reinforcement Learning](https://arxiv.org/abs/2307.04345v3)
- [The World Is Bigger! A Computationally-Embedded Perspective on the Big World Hypothesis](https://arxiv.org/pdf/2512.23419v1)
- [A Definition of Continual Reinforcement Learning](https://proceedings.neurips.cc/paper_files/paper/2023/hash/9d8cf1247786d6dfeefeeb53b8b5f6d7-Abstract.html)
- [The Alberta Plan for AI Research](https://ar5iv.labs.arxiv.org/html/2208.11173)

## 55. 2026-11-15 20:12 UTC — Mr. Xu

> Send an agent to work on this: "Reviewer: “Please audit the three briefs at 4dd0b9f7: `OPUS_TASK_E36_REMINDER_AFTER_A_LIFE_20261115.md`, `OPUS_TASK_E37_FRESH_REFERENCE_FOR_A_LIFE_FAMILY_20261115.md` and `OPUS_TASK_E38_WHAT_ERASES_THE_BIRTH_FAMILY_20261115.md`. Points to check:
>
> 1. E36: is the data call the birth’s stream with a different stream seed, and is the description of how the K lives saved their feedback matrix right?
> 2. E37: are the stream seeds the lives’ own, and is the claim that part 2 is batch-identical to a life’s first block correct?
> 3. E38: is the anchor code your device, do checks 2 and 3 suffice in place of a second implementation, and is the rehearsal stream built correctly?
> 4. All three: are the marks and reading tables complete and computable?”"

## 56. 2026-11-15 20:13 UTC — Reviewer

An agent is now auditing the three briefs (E36, E37, E38) at `4dd0b9f7` against the coordinator's four points, the memo's design and the code the briefs cite. It is still running.

I asked it for an audit the coordinator can apply as written, with a plain verdict per brief on whether it can go to workers and whether a re-audit is needed. I will relay the findings when it reports.

## 57. 2026-11-15 20:16 UTC — Mr. Xu

> 你可以自己决定实验的快慢，不用等我决定。因为此前目标出现错位，我们先讨论下一步如何做。我们坚持做持续学习。

## 58. 2026-11-15 20:19 UTC — Reviewer

下一步我建议先把目标换成一个可以直接计分的量，再围绕它重排实验。之前目标错位，是因为我们量的是"固定训练量之后，在固定题集上的成绩"，比的是规则，这两样都不是持续学习本身。

##### 目标的可检验表述

**设定。** 一个学习者，一条不回放的样本流，一生；内存固定，每个样本的计算量固定；不用完整的反向传播。任务族在流中到来、离开、隔一段时间再回来。世界有时给出答案，学习者由此学习；有时只提问，记分但不给反馈。只提问的题用的是从未给过答案的操作数对。

**一生得分。** 所有只提问的题的平均正确率。这一个数同时包含学新任务族的快慢、旧任务族回来时还剩多少、重新学会要多久，以及对没见过的题的泛化。它不需要划分训练、验证和测试阶段。

**要证明的。** 在没有参与调参的新世界上，学习者的一生得分高于同等预算的朴素反向传播，也高于同等内存和计算的"反向传播加重放"。前者是底线，后者才是有意义的门槛。

##### 四步

| 步 | 做什么 | 判断什么 |
|---|---|---|
| 1 | 建一个小世界（小模数、小网络、短生命）和一生得分的记分方式，先校准 | 小世界是否重现三个已知现象：能泛化；不在场的任务族丢掉模数；后来的任务族学得更快 |
| 2 | 把两条基线各自调好：朴素反向传播，反向传播加重放 | 基线的一生得分，以及重放占用计算的代价 |
| 3 | 候选：sweep 加三个控制量（哪些权重可以动，什么时候复习，哪些样本教得多），在调参世界上自动搜索，在新世界上比较 | 是否超过两条基线 |
| 4 | 把胜出的设置搬回全尺度确认 | 小世界的结论是否成立 |

之后再做您说的那一层：让这三个控制量取决于学习者自身的状态，例如每个权重有自己的步长。前四步里它们还是固定的数。

##### 我对快慢的决定

- **小世界里的扫描不再逐个写任务书、逐个审。** 协议（运行框架、记分、预算、统计）只审一次；改框架或提出全尺度的主张时才再审。
- **每个设置至少跑 8 到 16 个世界**，报告成对差值和区间，取代"三颗种子、相差 15 分"的标记。
- **预测照旧先登记**，一次扫描一行。
- **我用独立的 numpy 记分器抽查**保存下来的网络。

按我的估计，小世界里一个生命几分钟，一天上千个；这个数没有验证过，第 1 步会给出实测值。建框架是新代码，要工作者来写，这是最慢的一环，我这里不能训练。

##### 现有工作

- **E36、E37、E38：** 审完就放行。它们在全尺度上回答机制问题，结果决定第 3 步搜索的范围。E38 的第二阶段改到小世界里做。
- **E34 第二阶段：** 照常。
- **E35：** 跑完当前阶段即止。

##### 什么算失败

- 小世界不重现已知现象：放大到重现为止，代价是每个生命更贵。
- 在新世界上没有候选超过"反向传播加重放"：结论是在这个尺度上重放就是答案，我们的学习者与它持平。这可以报告，但不是您要的超越。
- 小世界里赢、全尺度不赢：不成立。

备忘录我先不改，等这次讨论定下来再写第 3 版。三份任务书的审查还在进行，结果出来我转给您。

## 59. 2026-11-15 20:23 UTC — Mr. Xu

> 但是这个还是不能回答问题，因为显然可能出现我们的学习者一开始比传统的学习者更慢，但是在持续学习过程中最后做到“后来居上”的问题。只要我们的学习者还在学习，而传统的学习者卡住了，就不能排除这一点。一天上千个是个好办法，但无法处理这种情况。

## 60. 2026-11-15 20:25 UTC — Reviewer

您说得对，这是我上一个方案的缺陷，要改的是量什么和生命多长。

##### 缺陷在哪里

- **一生平均会惩罚起步慢的学习者。** 前期落后被计入总分，后期领先补不回来。
- **短生命在交叉之前就结束了。** 一天上千个几分钟的生命，看到的只是开头。
- **按短生命筛选会把慢热的候选淘汰掉。** 第一份文献报告在进化规则一节提过这个"短视偏差"，建议生命要与使用期一样长。

##### 改三处

**1. 量趋势，不只量水平。**

- 得分按年龄画成曲线，比较后段的水平和斜率，不再取一生平均。
- 另外直接量"学习能力"随经历的变化：学会第 n 个新任务族需要多少道给了答案的题。传统学习者卡住，表现为这个数不再下降甚至上升；还在学习，表现为它持续下降。
- "后来居上"的判据是我们的下降得更快。交叉的年龄可以从两条趋势外推，再用足够长的生命去验证。

**2. 小世界的主要用途改为让很长的生命变得可行。**

- 按我的估计（未验证），小世界里一个到访一千次任务族的生命约四五个小时；同样的生命在现在的全尺度上要十天以上。
- 预算分两份：短生命用来校准和粗筛，少量长生命用来回答交叉问题。
- 长生命需要几百个不同的任务族，所以世界要能成批生成任务族（运算 × 模数，代码用两个字节）。

**3. 筛选规则改为水平或斜率任一领先就保留。**

- 保留下来的候选把生命长度加倍再比。
- 只有两种情况才停：次序连续两次加倍不变，或者双方都已走平。
- 任何否定结论都写明视界："到年龄 T 为止没有交叉，趋势也不指向交叉。"

##### 第一步改为先看传统学习者会不会卡住

这一步不需要候选：让朴素反向传播、反向传播加重放各过一个很长的生命，看它们学新任务族的代价是否随经历上升。

- 我们的记录里没有这方面的数据，最长的生命只有六到九个阶段。
- 第二份报告引的一项研究里，反向传播在连续两千个任务上从 89% 降到 77%，要到这个长度才看得出来。
- 如果传统学习者在我们的世界里不卡住，"后来居上"就只能靠我们学得更快，由趋势比较来判断。
- 如果它卡住，再看我们的学习者是否不卡。

##### 手里已经有一个这样的例子

E34 里按置信度缩放误差的那一组（种子 73129，12 个字母上的正确率）：

| 更新次数 | 4,096 | 8,192 | 12,288 | 16,384 | 20,480 | 24,576 |
|---|---|---|---|---|---|---|
| sweep 加缩放 | 2 | 6 | 17 | 78 | 73 | 99 |
| 普通反向传播 | 74 | 89 | 95 | 91 | 93 | 94 |

- 它落后了一万两千次更新，最后一行反超；在 40 个字母上是 99 对 85。
- 只有一颗种子，最后三行还在摆动，所以不能下结论。登记的终点指标取最后三行平均，把它判在了反向传播之下。
- 这个情形该做的是跑得更长，不是跑得更多。我决定在 E34 第二阶段里加一项：这两组在种子 73129 上各跑四倍长度，约 6.6 CPU 小时（我的估计）。

##### 边界

有限的实验排除不了视界之外的交叉。能做到的是让结论带着视界和趋势，并用加倍来决定视界够不够长。

这些改动我会在三份任务书的审查结果出来后，一并写进备忘录第 3 版。

## 61. 2026-11-15 20:37 UTC — Reviewer

审查完成：三份任务书的事实都对，但都不能原样发给工作者，一共要改六处。六处都写成了可以直接照改的文字，改完不需要再审。

##### 审查结果

| 任务书 | 必改 |
|---|---|
| E36（一生之后的重教） | 三处：K 组的代码路径和检查 3；检查 4；判定标准 |
| E37（全新网络作参照） | 一处：补一项对更新路径的检查 |
| E38（什么抹掉了出生任务族） | 两处："丢失"的标准；对锚的检查补两行 |

- **事实都成立。** 数据流种子、K 组保存的反馈矩阵、锚的代码、复习流的构造都对。"第二部分与生命第一个区块逐批相同"的说法，重建了九个生命的批次并与回执的哈希一致。
- **有一处是真实的代码缺陷。** 默认分支上 K 组的运行器导入的是另一位工作者的同名模块，函数签名不同，原样运行会绑错模块。
- **有两处是纠正我自己的错误。** E36 的"比参照高 30 分"和 E38 的"低于 50 算丢失"都是我在备忘录里提的标准，都不成立。原因相同：只丢掉回绕、其余都保留的网络，在留出题上仍有约 50 分。重教在开始前就有 13 到 37 分，"丢失"的线又正好压在要区分的情形上。两处现在都改为只看需要回绕的题。

审查由代理完成，我自己复核了导入问题、两个基准哈希和标准问题，并重算了下面的表。批次重建那一项约四分钟，我没有重跑。

##### 对我之前数字的一处更正

备忘录表 3.3 原来只数两位数字，没有数结束符，而工作者的记分器是数的。按完整答案重算：

- 需要回绕的题答对 0.0–1.3%（我之前说 0–2%）。
- 不需要回绕的题答对 2.5–93%（我之前说 10–93%）。

结论不变：丢的是旧任务族的模数。

##### 备忘录第 3 版

已推送（`823a3884`），新增第二部分，就是我们刚讨论的方案：

- 改量趋势：得分随年龄的曲线、后段水平和斜率、学会第 n 个新任务的代价。
- 生命长度按四倍递增（64、256、1,024 次到访），两个学习者的次序连续两个长度不变才下结论，每个结论都写明视界。
- 第一项工作是看传统学习者在长生命里会不会卡住，这一步不需要候选。
- 小世界里只审一次协议，之后的扫描不再逐个写任务书、逐个审。
- E34 加两个四倍长度的出生（缩放那一组和普通反向传播），不需要新代码。

这一版要等您转给协调者才生效；第二部分如果您有不同意见，可以先告诉我再转。

转给协调者的话："Memo revision 3 and the audit are on `claude/vibrant-cerf-sje726` at `823a3884`. Apply the six required changes in `AUDIT_E36_E37_E38_BRIEFS_20261115.md` as written and release E36–E38; no re-audit. Items 6 and 7 of the memo's section 1 are the new requests."

## 62. 2026-11-15 20:42 UTC — Mr. Xu

> 现在元问题已经非常清楚，如何在有限的算力和时间限制下设计实验环境和目标，进行有效的持续学习实验迭代，平衡短周期能够发现更多样化信息和长周期得到真实重要反馈的问题？考虑到后者在技术上也是一个RL和agent设计当中非常常见的问题，你有什么思考吗？

## 63. 2026-11-15 20:45 UTC — Reviewer

我的看法是：不要把短实验当成长实验的缩小版。短实验用来测量学习者的几条"规律"，长实验用来检验这些规律合起来能否预测长期行为。这样两者是分工，不是在"多"和"真"之间取舍。

##### 强化学习里的办法，对应到我们这里

| 强化学习里怎么处理长延迟 | 我们这里的对应 |
|---|---|
| **值函数：** 不等最终回报，学一个早期就能读出、能预测长期回报的量 | 两个"探针"，在任意年龄都能短时间测出：学一组固定的新任务要多少题（学习能力）；缺席已久的任务还剩多少（保持）。长生命用来检验它们是否预测后期水平 |
| **基于模型：** 学会环境的规律，再推演长期后果 | 三条速率规律：缺席多久丢多少，学一个新任务要多少题，提醒一次恢复多少。E36、E37、E38 正在全尺度上测这三条。每个长生命之前先按规律预测并存档，事后计分 |
| **多保真度搜索和种群训练** | 先测短生命的排名与长生命的排名有多一致，再决定短生命能不能用来筛选；晋级看改进速度，不只看当前水平 |

第三行的毛病是已知的。有研究表明，按短视界选出的学习率可以小几个数量级，即使视界有一百到一千步；种群训练的选择是贪心的，后来有人改用"改进速度"作为适应度来修正。这正是您说的"后来居上"问题。

##### 环境和流程上的四点

1. **一条长生命里已经包含上千个短实验。** 每次到访都是一次学习新任务或找回旧任务的小实验，所以长生命要密集记录，不能只看终点。
2. **不重启，分叉。** 少数长生命作主干，按年龄翻倍保存完整状态。从任一存档出发，只改一处、用同一段后续样本流继续跑一小段。两个分叉逐题可比，差别只来自那一处改动。这样用一条长生命的代价，换来大量"老年状态下"的多样短实验。
3. **时间用学习者自己的单位。** 按现在的设置，没有被当前数据支撑的权重，衰减时间常数是四千次更新，缺席长短应以它为尺。另外要检验"变老"的年龄是否随网络容量缩放：若是，更窄的网络更早进入长期状态，长视界就便宜了；若不是，宽度只是成本设置。
4. **对我们自己也记分。** 每次长实验之前，先凭短实验的证据写下预测。预测准不准，就是短周期对我们有没有信息量的直接度量。

##### 走不了捷径的部分

- 需要从出生起就生效的设置，或者一旦陷入就回不来的状态，分叉看不出来。通过筛选的候选仍要各自跑完整的长生命。
- 领先指标一旦被用来筛选，就可能被优化得好看而无实效，所以要定期用完整长度的生命复核。

有两件事现在不知道，都由"传统学习者会不会卡住"那一步回答：短生命和长生命的排名在我们的世界里是否一致，以及年龄是否随容量缩放。

这些已作为第 17.10 节写进备忘录第 4 版（`ef2c03a3`），只是给协议任务书加了几条要求，其他不变。

Sources:
- [Population Based Training of Neural Networks](https://arxiv.org/pdf/1711.09846)
- [Faster Improvement Rate Population Based Training](https://arxiv.org/pdf/2109.13800)
- [Understanding Short-Horizon Bias in Stochastic Meta-Optimization](https://arxiv.org/pdf/1803.02021)
