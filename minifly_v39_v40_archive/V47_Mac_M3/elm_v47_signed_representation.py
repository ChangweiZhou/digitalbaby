#!/usr/bin/env python3
"""V47 — Signed Predictive Representation Learning.

Locked scientific question
--------------------------
Can a bounded online representation learn both what should be invariant and what
must remain distinct, so the same fixed-size, slot-free global distributed slow
memory avoids the cross-identity interference exposed by V45/V46?

The persistent substrate is deliberately unchanged from V45's GLOBAL-PAIR arm:
BASE, FAST, one 1024x256 SLOW matrix per arm, 51 active slow units/write, the
same BASE-miss write schedule, and the same 50:50 FAST/SLOW deployed reader.
Only the frozen representation differs.

Mac M3 quick start
------------------
  python3 elm_v47_signed_representation.py --self-test
  caffeinate -dimsu python3 elm_v47_signed_representation.py --stage all \
      --workers 4 --out-dir v47_results

Rerunning the same command reuses verified completed jobs and resumes partial
jobs from atomic checkpoints.
"""
from __future__ import annotations

import argparse, base64, collections, concurrent.futures, csv, gzip, hashlib, json
import math, multiprocessing as mp, os, pickle, platform, resource, shutil
import statistics, sys, tempfile, time, types, zipfile, zlib
from dataclasses import dataclass, field
from pathlib import Path

for _k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_k]='1'

import numpy as np
try:
    import scipy
    from scipy.stats import t as student_t
except Exception:
    scipy=None; student_t=None

SOURCE=Path(__file__).resolve()
# Scientific identity is pinned to the pre-hotfix runner. Reporting/serialization hotfixes
# must not invalidate already completed scientific job caches.
SCIENTIFIC_SOURCE_ID='4f1ee4cde7c33102d6eb0a2a3eaa3060234e15f532a76638ec6756f7fd13836f'
VERSION='v47.1.0-locked'
DESIGN_SHA256='e55645a1ee2fec2fa52174709a92374631a23b8f069aa75b94578be190514054'
SCIENCE_SEEDS=tuple(range(47701,47717))
ENGINEERING_SEEDS=(47971,47972)
PRIMARY_LOAD=256
SECONDARY_LOAD=128
REPEATS=8
OUTPUTS=256
D=96

# Persistent substrate — V45 global distributed reference, unchanged.
SLOW_UNITS=1024
SLOW_K=51
SLOW_CONTACTS=12
SLOW_RATE=.6
FAST_UNITS=4096
FAST_K=205
FAST_CONTACTS=12
FAST_RATE=.6
STATE_CAP_BYTES=int(3.75*1024*1024)
INTERFERENCE_MAX=128
INTERFERENCE_PROBES=48

# Stage-P positive development.
POS_IDENTITIES=192
POS_REPEATS=2
POS_WRAPPERS=(0,1,2,3)
POS_CAP=1024
POS_TOPK=12
POS_TAU=.58
POS_RANK=20
POS_ETA=.06

# Stage-N negative development.
NEG_IDENTITIES=128
NEG_REPEATS=2
NEG_WRAPPERS=(0,1,2,3)
NEG_CAP=1024
NEG_TOPK=12
NEG_TAU=.70
SHORT_ALPHA=1.
SHORT_SURPRISE=4.0
PROTECT_RANK=8

TRAIN_WRAPPERS=(0,1)
HELDOUT_WRAPPERS=(4,5)

ARMS=(
    'RAW',
    'PAIR20',
    'PAIR-RANDOM-PROTECT8',
    'SIGNED-NOGATE',
    'SIGNED-SHUFFLED-NEG',
    'SIGNED-GATED',
)
PRIMARY_ARM='SIGNED-GATED'
PAIR_ARM='PAIR20'

# Embedded validated V45 implementation. We reuse its stream generator, BASE/FAST
# substrate, Candidate-C positive verifier/Oja learner, and numerical helpers.
_V45_PAYLOAD='''c-ox0*V639vL<%Fo?^{RNZb@i!n>n5lHg6lH~eUV0o#p+_a-JL^bmSNJ&Euya&zx}bf^CUT~)|RRAqjd`Q;-2%YOpbc>>d_1B-6^KAg+D@6d1m`9J?>AH%=@KmXVN{axv{WU7)bD}I&MDlhWyd^Imqm9EP@=K5~#n<`J1s_%aL_D$=v`d%K_V*XnB_Upg?edChu`&6uRk~YQnrbwplmahK##QiQ`@1Lq$im5E7;%lww`t|kxUiQ=X<a=2iMgHqtUBxfo=B8hMEvKURp8wO4o|+kaZ;Q5{&R@PKUH(1qi}`z&+#hYeXZ<wwX+Qlq@AoAC`gmRp#r^1(Z(Gu=#rOW}qs#m2FvC>L_lrNItNDA?4pYAs`Tzdyn|lk)S#`gy$}0O(TzpCVhdqCDnBe<1f43OL`PEdJpT8g9^L6^V{;&H~Enhm`_nn*K+bwGFrv9tWxSag8hcg|=WKD;01^(w>zVFrY{{ECbQPc$f*Dr-XZujS|r^h}mh7<0Cbj#E?^Kaj*VrrA_KFaqmL(SilVYm%`U#~g8C-?Mw*LS~;YtmHZS#_`564~0OU-DD&r|oZ#>PhxpM{kc-S^a%pl4<#mPW|%h_rF8ud(pJNZy5gjKb-pe>j!Rr&;A7~!}<N!U+1DJe_d{@+zPU!EQ_j3Zkzm?SM9w1|5Ujoui|^sG~fTvTj|%|ufO)|a;u!ae<``Wvo_28?VD3fYu9~U{L9yK(iXSv+V;z)UkJT1vlY|r+Z$5%`p^_#;QI9G{QVD+d2Mf*;WkKh1MBO>w{q&+@5$1)Ux0WQSv~YOl5bbs@$|>b-+wRHWu1!OfB(W#KP|ViuKxop_nH2_Hob1WbMg0in#=`({e718O;col?D%Wx_T6;5Z29dE{r~i4IR6^Y+uw(;s~Y}3ekb$0snWksKl2^8e~#N^dH*%O%~sv;&x(2evf2%y_Aed%UmqN9oHT!(d}xy84&3(d^ZR7|I%R*opWo|JHGj>%3~?up>@RKe`PXnM+M%rep?ay>e~i!JA6LGw_Rn(H*WG>?|AtADHOc(t+dp6awe-uKS&AnALvXmSeee3ako9GSZ~x^xGu$Zn^;tF5ZSU{k_r0I$`OiSVGf-9yP1P0u_JxTXe?@aAsqaZP_4E9FE{Ys%`s5D;{c?xi@}1R~=WpM>(%SEJao*wg{ck^+mSB9X|898t@2>0!4FCM;m+wF65A8oLnY#6_OMk-s&%d91fzuTBcby3p{a<(9yM)kk{$=;i|Mp{hKmT^ufokfz|N3>P|MvagzyAdP^cOI@wH<yWk8b$(ZJEwLHvQdKw*k%{3qOqbWBS{deg3lc^0jqd)~{sy`|_<gvSL`i3qO8tBmSZ7$G-c!M&A{GEc|`L-xt3<P0zVg-M|0v%-`;qy50Tz??3<cg|22R{`T`XUvQp^X?z|34*K8V-=JTcAEoK1Z@lnSB;)s|%pf@NB>+K13MCMn!W3FkC;?$PjZ#^H<rqR1WmYCAi6n3iAqY*9B$8!C4iQLE;t-W$8THdQ?#{sqf9L7nnSattr3IN|P)=hdf)RO2lz9SS1fFG74&!N>qcjHNB%0n|B+-J-@)E@<G(j+kAn_9YXT97%pSMh)5=PUsObfb1XgsB{1S@HnEHM(NNf?9Zf<Tfa!HXn8aTF)u5=GDyLTMO5pr5`yIYlry;rFK?@XuT2vfjY{+mv*#;%^v^L+~%(?;n`_`R8wfAt{C+IEo?tC4S3b^bZ+~Mt=TgIf~9W(eIkc@b}yMl;55LXBzxJEy4)(%@HhtalaH$-%QW7ydUE)@h=IQ_;#1F$JYw<&2{cl_B+Q|_Y{YbKj$vv-Sgil{M%QJ|J^f`9~xi?!Twlz{1E?Rd>_DNxIYAtKbF2K>6ZfHo5(!fzxLne`g7%vuj7AA{#C(`rGF?;3|DYO!TDjvJH-CH;=e3gcl8#2e1Q7)bbok}v3_SG_s8O082*O-M&jT9mme4S^XKQUJadzz{HvxFzjFVPPka@;CAn+LFW>*b^Ihg|Ch+?|O!s3dxXjl%A>xnepN7M!Kc}Yk_~TZ*;Q#*5w*FztuSJCy3|DcLA4i4hKP3O*z(3}J!&ufGXkU<ke))bQ0{P`T`tvtg(D>WI|MDLD<vafKmmd}5|FiGsZ?~;QP2t?%#rrVsAA$QDjNmu61X202Zu-;z`0xMYzyEaWX81q1ra%23cQ#9Z`Y%78|ND>UfBNzKpME_5=bwJQfoL(e*UZ1~__rT#{{6@EfBW(LUw=HoA5Y-k@erclL({pP`h5J@^f%~$pkKZoKmT%%VfY>+$d56K-eVm9F^0r@jB<a-AYWtb#~AsN$Iv^JSmg^;{1<|d-)?_*!*6RhBw3wS%lx-{mi!TccSQX1J^w0c|NbR%1BLqezy8}VKXTuHyD9Ba&HoPn{97`AiTtg+tIG28_g}xmHxvX%_im*9VZeLytuXJ~-}0hN?&|&f)V=Q5MIq?^#LvGCH$Pnez}43`Z|E_<TTJw4#QszmW4+RaiA!Umg>mWPLK2oP6y;efmCgmROvI}yOA^p4HCR#r-TGxS^jpPk_f%W5AY$@kT>0@3$NF;`IGrabs!SG0m%9}fw)Poc#Z_)|auHrutggI)m$qH3)4}nuNKQ!`B}juHwZWjO3ovg97g~ePWNk|yKBl@{Cd&p_dU{U;ijq|SDC$X#<{*y-032teyl7A_EfAQd+gm&gt8ECZ86IrpkxK#Zk0;(C4eX}|O%huXBS<o@z_^{|L;<EAuj3?us}EYC)r4BTdEnS#@isoTjUP|i%(F0WXh=JBKfM5bs&je+_5voPN4aS*3TLf{pkkU0*yTVwSUxO@#4{gsoDdB(nC3%+D#{qxL6!o`<<pc!N#iv31QQd@F0~0tAe>M<+M3;4X`tIKTn=F!SsLk?mw9Ya85ztSzT~F^6cBKqY$Rzu*KQtqr3@%#5LD9W<;UdVKSC6)h!JRj&<~L_-z3!X%^;>tltxQtDxT&JYs~`DLsRF|N?o9zzClncwo;#3s@=@Fa{>D!M6V7c?}{TID;;Y6$Y}il;F(T<`T0TcgKpIUsf-6$sv;U`8osSYcvH6hIe}Y1;UoeU9AGap2r8mQRba^@J4u=V1irT%JwPo2T54a#hjrZ^Dzyd-F~m#7Yj*%tZR!1*P3k~b<1=1FPDG%*v87Y88;Cf^NkzrTf+UlGy7lfF>wLhULxS*&p35*ie2`!JYa+`91n6mn=4wi<5uMO?1b{Zckn?saI7!giiH2>op%z3t-~gtJoWs}ls8{d;Kk`1b=fgrs!}&nz@qxl;bBJQO&^Uoz1xA7Bwt(A*<t^ys3mA+|S)K?Nfav>%L3(Ky0KO;erWN!`(;1#@!26o0R56wn0@AW<KkXA{2OWIjU<8(txo_Jkf`E;J8%k4PQf%9SQU`^RbPXoa1Q5xD=bs@WL#VsRoXIh8(Q7y^+f8-*0BymnT+8Xh8)?c*Ib4AGTq@~m5QNe^3fe8f?jW!<JvYxpS)`l@J0q*h1Tk+3{Mm;?)YeV|DBs&*panp7ZoAWN?$EU4hGp^w2vnxi6mzS%%ptlr#yJJ4E(4PO1c^OLT4Jm~Mvccusx>)4rk@nYK@4grZE3(LqqfYby8a0ONBb_o*#soY3`p&hQW+tGrJh>?`li$DW<Q&`Vk`hweJ*ga(~%%S9Uvi=pS8QAlajy>2~5~D6r`R9<nf5dR>P-h9kzl5KS+cw8ucOG_NPNhxdm{y+&0{0OdK}u7|KLB56aKs(lW+ANnZv?;4!j2F%cErCm29o<iP`Hu2>TEh(Q{lrnQx8_{z~D$E0?hw@AZ2w^cn5qfAgm0nh;tdMm&kNH=i5Pz>x~3b-4(va{0bLxva8OZc>HobPmP?t?odShP8|bO|njaqFwBHpG{q1Hiu6L&w7Q8R?di@SC&6xOSsV38+!sx+rI^Hz?P<2^|#@CUBr+<#Mxl7yv*Vm{DW{0RS%jPi8<48Gv?)S*dj$G?crPy_V!W>iD{+GH$-p07w84hsw*?O0XV#vYAKlwS}NZp%3Sv;Sb&r;QP+!H`xLc*Cef+kjwXZb$|lLFo{z9X{<Eys6R!#%^;27DFp(pB^E4=Y9M%@00ro|uvVY|$`)MfZ6n}|9?WqD;(Q{)Fz`?SL2rX^1Vn>CtXF3Vx&nkcAdz+2yea!;pzksg@t7`9eTl-W1e3BYK>!ZboxD=AphlByfx9%MAhB&Arh<!AMs%>DG19J=OHx^8HJ7*8{#`o8{El_Uy2}O_Z&Qa}7ER&Q`L)EClgmTP<tiynki`X}Q#S_FrdcSyT-(PcZ$N`nCm*S{jVGwBTmAxRqK60vEapwxF3s5IX9<!Q54WUNlG8?$I4~wx3%dZQ8+9Hh$dm$43ng#2QTkMZFx%^;v7!|1W>0_+@MR@4k^}}4pZT)OJ**^=7SsUXq@RF@62xwC;FjVmDGqptgB{2jU`VH+2Ol5^g;IV_`W?_RNFS7&-$jFb8rv44APeG{dZ%Wb1W!gFq(~ri3JevO2$O<O;S#v*qk}P<F2hd2X?OZ-gd89Ep%4Q3I->OHRIjCLB7SdNIC!hAOBPJgsmAO0SZF6a9*$JMp{!9O0zDr3m3l=1KJ%^s4cBCLOOlpA3|Foq%tvY)Mr(Q`0t~Jq*2p6mKmo^gV<Y)KEyXmzTbTm97t0uLHcllsc!P#AK{e!;@^d8M0i9kb{-i6t3hfpGbAL_VFNoosHZnXdl#wgC4y6leL6!YuCeO!{NIJNyZJ;Db<}m=YX!xOXg8RT+O`rr6a%GXsYT8V`eIJ|{z)L^9B-V1BT%%If&tZKv8$>8COX<+{;XUC|>|guKIfu`kr$)7*TvDEo?y7U#HXT(sUF0rM?^e>TzRTh{v2i+NePOPmO?&DixNZ$Hnw>qIWQv8ByaZG{Kd4L>$vEYo47FB?^oH)t(&vm_QoLDZX54vexhwJYM(<K0LHTu;FKM2$=T!27nfjf75cXcymdo4K>!w+RjYARe#1AQdBpoY~NkQ$K^?~&y3w?vosx!@L{_sBdC7QZ?waJJGD{?opbf0}DF{m`jWWtV_+e=mYNNJTrQ{bDxArN0_%`ihZu0;v2X_aIJyT2lbofVoF1Qm+r9KNYoSl1!JXHqq>#pIA`EHHj=*{bvGS0-Gvj+3<q397hh+e-RSw9_s!y=FwF62pFTWt9|w!yLjpWAW=-84WNCBNY)U7$}B0U~YibAU&ym2D&<2GQ6k2ub#cTl}~!${=$2{&8F%~_ezItZ!EUy2eCI1<Rnoo39rj7_BN|lh0|@7r(T=@!S)KGU>68CQI+@G<y}=#wLQkv0YojwjO39?wJngbhd?Z8L)Ub_8eD)`^OdXRsRo`d3h@IyzF7#IJkXXBy~do1fw<5<HjaN3g9L-<+h)8mdDUWP*28^Ot~;KN@bOrKwihG%h?Vz#*BxR;@YO|3K!@qc=2Qhi6b#9B+H<pf-6<h%Pf+rKO{8Bn--&F^7fl<#LO^;0Q}m`EpMBDSO-QU7G{v`of-TX^9shPsT0VGN(W+usAHlneN-N;Uf^uF4b$&$S;sJ6mFny17!s%Wt>th=eYrfdAQ*W{G$mt6y?S`A%F6*#@>XK1`yiSTJ^?I=~;M}{DoK)hbzUiXv?VW(+LY=&JHFD3mfoji-T3R7LVOpjerT0+X7MCa;V(Hjhovp2*uyYWbi1@*E4pq$?;hfqo(!<6xiV$`ncDB`;_tN@}(Xh)4iJp8sMWf&Fba@Z27njCoB47pdcUEs^CW}-2flJny5c746wk5ZCqx2?09|(_c>TK%={V|_<X^X?I4H-YEjGJMaKYQoAstGlFJUqt{mcvf$#V098!@$o&n;*G79ZGa|BIbqxB0PQhv&B1#DO9GVC?Vd(K^md_SYa4u>F92X@nwrH95|&<uDtg)(eAQ==Z-3I3Ntd1V$F_3B$moHcGae~vPwj?J)EcSn0H54+(o(u^T``xrR#Fdw7q5!RX1p#RJIIRrHJzA*z0RBk?Rp*6e$!zaH8g$5ayFWiFuiTaQ6%@r)5ZuMF22%@1|=G<2`$kAxUr^st|gJy8O_34HEYi3(RC`qQSWxmba{~Kcb!u9rz}c?@@Ie7wem336KRihK=g1N-)mt73A-fKG1X4*S+>L^<rypRE6;f!9pz~)h>C9Ii{PVd;8RqpR|Wf!iI1YErmxgN8PF+AEtpDUElzd!8Y7dQvi2ZKrm-P8&mkErIrq`w%Xy^h2Jq6B6oWI6tXg({pNAW$1EG<iNj9)<b8&`+-_)=iG<(DX^t!;LuviY3)z&+LYye#p;FXqr)rPA;9)a-I(Z?)6*GYW3|d1`gKx@OXdv#Rv;}DuiLF*2b>Y!j<1C(so<~Kge8fwsa#kMlQE-Wl#xUh6@!{%^8he?=EAnQ;ro<Q4hM|bV_Oi9j;O$yFVBLeW&?&1`hr5}15gv`?S@lg-vhI<va2TWNeR^FpI`?u*2v-$9TIt+;&^BxNLuH0u&gUQDK>)6EwX<YCQ;bO4-fQj#DM8lF!3iW&WQgrsW=z^hqM2(59bciv70;q~)z|yrzgRED!B%)DERFFQAeG!~ofTolGBWziHZLLXCM~uFb6KlGD&IXL<9EbPsE4gBkjaf`V~aAR&$3tYH=jhh>m1%O5opp`z*!0<`1>=N`2mNU^a%@>P71DLpjo`0lA#o?B%ukE2FDN@7@q3-3dEtcd6l)jYi-0;r1JI7m2DL&9ZU<m;AO1GQe^n!z2htBK2Z=ACc%1<p<^6jd8lrfXb}49Q{s3BLss&Kbm(CWgAD4R_VeLlA57z&7uTNkbXU^z91<t<gU7Tb-4z{^SB<<MoVRI_;1%mv=20r_g;wkvHPXi5{B*Bpi;|z(e38%Bh88~eBIr18E5%UlkKd=bMwJx}_g)_-*YZA<F@s1nI9a?%)N%#jv01c(k>;?;>ikUgepbl*FlHSj#)d-4<EG;TLr~A%M*}j;$aqg5iKeo}_S9a6m~s~q^h)NNC0;vZ)kS!Z#RQQZ@QJEm<5K-=+~f?e!)NC#2hA@N+>TE}evR78(TBo{9!$)m7f-JOu?Nd&#7HRL?d8(}H<c6mN2E-x485OP!6mv?uc{W+xY2L{#c|^xkkb>!_l3n}_2S-Q3ILGRS8c|yLh73td7g6DR>fziABgvQF?1onxnkql2IZI6E~ISYRT5-x41EKDV<z#P9&pchj`SEMsHsp7OS4!}g(I-!=}I|mR(y^PBaV{BA>MfvR4e06^#TWGQHa;7o=L!lsA-D?Nl#AKY<ce?67IvkOvMn<mwb-9GInZfej;Qm;!)%(OFC&xy_p)r$mc`_UGnUe3jXAU;5yNuz$pM`oFCao*9}Jn9?AUpRBX?@^~d#MNLXR!l8F(a6z3^dBF(E{X<icRX>lY(U+iPVng(+5CqLyZ&_9SmxIDJ5Lhqx39Y<`~=VXwD@pR}J)tQ<0&!SeP?fnYwtkeNFuC4sc8H>0ep&$k2Fb~j+bpk^8d7pjD&_g6!EL*($J3wsA)k;|fFH^Q^$mmI|h2t=kAs8i#gS*RRc3xw4szs-+JwLCToh^qey%@w?%5e_HwR;z}>$c^it_{@GJtg7UI#qJ&t#(dwd3Z*62mKNjZsFx-+8$=>el#n^Psbo1OiA3#eJ>8T85(a$lEZR>a?yKlk!mdRAChv{gBWHhj=nP-jhqkrlJ~km2x5D{h7U}KggIcI6m;V4qBVK@OPPxF*~0iNyaEOHN=ItB2YFTJggqZ?Z!eCkJ;^o3PM1z|KaP0(qt%tzAx~(VUOsG2jFw&;b661@UZ~$O&3f>9p1ApirMj!TWLQy8lW==!luwi0?vhyYaBMR+5;A28v+MjSk^=V4Qvp8K#Sine?uhP#xikSUg1L?%>(Q;0{FmQ!i(xf^&$<(*kd`HuhW5az%J)<BnDF>TxT!23*PTgHrJNr+xxB!ye<tqs;CXe6Jh=&`4xg_!&pAVS8fw>;yz2uIGV%Z)^0E*)k-QA=t-q%_g4~C?QHEddJI+hv$<yp%X*f1#`Ppztyo;Jmd*CB%Gwrc1iYO*u<cdnp4^;xm!=*;UepD4(j}?T&mf}pRD7GxDUotZw!*P_nEIM@aY$5gdwEMhgkjGZj&AOhLN@@!hhL!|G>Vlj~10#<F683&71l+(F5(-QtZs0Xj{pn8qK<(rFb>+zTT0Uyo-aK;BC+v82u4T8kHB;H7R$qSY1zib4!kfx2GS8|bBp-87R$29jnaGgw<VnL?uJjiE^o~--&(6z&c<z+XT@QeIcX<ij$?JM#;e(*M(oH)xu>0nU>3y*pryGmeDt26D-w3WxkAifBJzrB3OYMt_jHh*drX2|e<0PS-@EHJq>>CfM;%8Th)H=v=E+W;<)Tg|VtL`&=0zEMP1(AdcWBEdfY|v1^C_o8g*tiYs<@yS#<8Bz`E|RQMA@_O<9w94J4<cUsun7ks6x2=Q-+pXBIPoNa%+@%<X`mWyM-;!=9^9uO1lOo=M`6gp6Jmt3X9V;7;f=XDCTPSK_A?mSUcO(Mo~5yZ`%m+F=EM`t*8)LwO*>46DXtTTsVDoc*R=KvaqkYu9U#?{ff&=9+Vcos2vM6b1>!>|T{QOWYj!6S&yNZbUuv*HLPFFC?(Pc*9YsMYaN?^Kj+`N^WoA=o0{)&<?Dc|Z<&)3yA?CK_#dIO%ElSVv>y!pliNUEvbopsh?yG@lQL^d)@k&#B(``TzYyTOO&&HyQdH1P4asa&6->hi^-uxT28IY)#v8&vVXg*$dntV@*XJ=v)r2#^_j}(bGfH~XrBsKrM(z22xf+suU4e_Cqk<lw~-4R1&A%4oANVgwIXZLVjOPI)yr>7Z^bR1za^qRITvb;Rz&d74E0ff6hTZY-L4LGaq6*sY=(&IpCEX)H6N?uiH0Q_j{BYdt!A)FWNshAW<lrNxPAlTunkEpmo-^>pzOAAQP=ZE4_avh`ws+^P09ob`|&MYiG{PkFf08_Q}Shzu_U^wyMvFs*{G-k8PDvD^Iq|D}XVY~3u83&CDg+Opn<a2qI5A^4C6KUOd@us1}jr`c^`Cbp#kYM_7F7oD^s&-eH2G9dGC@`QwG+!RV-uJESYTm0eTO3ni2YF@ovBR${=RHH3Y%_*}pn;=U&Nk%uR65`4I(M{`@uDaL>uu6`-u?aXQ2dtkd3cEq|613Oe@W?)5EorLwoxUcnk4yAoLeXsZDa=+0BQYhO}Pa#ew%=wIgxVokmq)(x#l5^T4`K0&}tmFXLPVY{dzI|(DQ-At2^ypLvB2ePVW}z>H<f;On}~{R`g{Tjur2P9Ri0pNd|9PpKj>OM@;Upk%uWMRT(}@4lzJP*d8xdFa6M_&fMP|`w+F*vfThIhnHeKK2h)>ZsKT=h_ZjSp&Qt{2y;Fp;OFFoKp`dJV>;sLsSlu@*F5!lSH`(2#5<8O5nF?>)J`0wp<3n*YMNaMxw3Ru@kAe=1<XGC)~mvgpddqn2&{XzF3axrbilJ|E0>h&JZeY77RHvAajk<QlC3M9cFu)L7W3Ye@7#Sc+L~DWHnL)ncfb3*^B|`LiG@v1D$VZmsy@<dk(fc(<hffy4Vegh4lj|zp+=j_#?`y>rBTH|urD4Zk|Y6&F&WRns;YPCwj6L4#M2^FVBjQ?TMDx13OmaMJPTiU#~H79^hzhszyeJr-%$z|&YPa{s6Se7>k9+t3jP)F-4Pw6NuOAx5vVr?q<o$4EHfrky!SSUVidruwf2FoUj~&uLtAd=PZCu8lU7C9o@iPVk=K-jDi3EH2hGQP^`<u^Z+SLA^KM{cW<fdg5}wIN(_AvJBgKXp{w7&RT7YBr+Nz$iF9jF<(MhC372_&#2>xBzD>`862Z`S-N~<=!!Z%%(SZuf+<AcHjXL=neMU@Nyx4}`V3}89qg}e!u^5(Y!J=s{3Rn|f|hf`iw)yz+Fu2I1Dpx4-+O~G6XCT<L}wjXk4L)juQ3vPQCEe_n0Z0d7Vy>29|7DiTBib_-kF(>GZq>{Cynuo?cOHiyu9gcC$THYNfWTL|1dV8n9(cvtnvHINxmfdcEoGT20y1UFMKb@`5AI2vxq?>hMuanM(bb6Ttg(k%-*?HM`8aP6|5B7vMq{%nG?xyh>wy(#WN(i)P2?tR*clXOvG<+hES<dwBuEJ=m0dqx@h@xdk*6atef>@3RQ%<pXN;0yg+(*W_QqnhA(A^xz<37ewmVL|<m!Q85j%H5o3cHLDi<5&NGnXMpJ+hQ~R<PD?QK@yD=UXkEh2`aucqdziryoR;#&>kW)=5ykOVeC6a?F+)`_pf?6rh5_4!Tfui?fWLjYVp2uPq+?5nfdcY9467n_9G=o3TRmT>uN?e4<N?VXqHyo7Wx;2BI8v&yZy_fxEMc`IW>Q?Z8dujgKgqhtez8$R}G!MFZVk(Z+{$)zo((Zf<_RQxWMft~ZX*qXyo_l^rvSGu0vws)D*=Gc_ALH?GXtU879exie!uD{=6!bKt0z9k)JWJ6Jbkg+g4iyY<CP`(#ylMGjf5&b2A4IaN{x=c^*Ham<?p`ddy`==HQeuny@d&63_n7nIPK9S4#sn#)To<a|kQWzrLBS=#~8hjIH1#@^wU7xUmA`*os=`mFOZT)AO&ssS|w=_5g}wly`xc)hCm&EXxS;<V9!!p6XNT&XurvDaXF@EJVgY87&2kxZW4H52fmawbasIs}@-$pKawScLnakBYTn;<Msu@-+NiszfPK`~}aUOxvs1WYH0`q5?%m?})-l3aOaa4Od-`Yx5Yj8&Rd;FJT_h5<ZNiv{^~C&Sx0MabQB4j9iaAdjvl1vXVlG(Fo!l0tH^+(Sj(2-Q~Aus<h*BEPm1}WPh<$XI@fsbv>n@gQ1=#|3(rl19n#_tewIpm@v2#AP-1@+<J4OOur8LlquqjSk@ovi7U;p;^P6|9}c?sO)I+@#A5mCSINE!mCVHHyL!&V&Y4dQrManVWRk+y;&^C}Oo&~ur}0;Gx0Cb-n_4}f=rEonJKDKLblTh_;Pq+&Z?`<*ynC)L<Y)19`c<R8J{5lFz*on;vawf6VS2V+a2jgBS~fIWiOjx?X|42XV*pvk2W+XCsTI6{hcQq-Q58`J_XIQ=%5-VJrbu~)Re;tOQlpFuvzeEPEPxHuFx)lanQ5+woRqKcxfFeZ78sr{Le{#E^@2ZmOc}Fat<`%OlsA#ZdmHmRVBNXg$GVq#l`oGAZpv?zz?k3VSa;tTS#jPJp1@+5tY#n<m!G85j1wG|Xiqu34f+qvKoiPzjRL}=J+&@vd)&3!<L$6x9cfO*9F!i=NlZ_inWoWlUA!<5@?9N(|E9G{m^<?Nyo6bPn;|BlnXgb!fhE#4yXl|lb%Ybg7v?d3mZ65jURhau(j$uQHW=)z!V*{ql?d!<6`KEi!VDLPP-x>LrxP#T65s6R0ZFw3?OTV|TZF2IB8=}+R7bFE!yuo8p7H=ABh2bl4Y;t5#!m_Nx&AN?GPmjE3fx+UnV_EaZaAlGJ_8pW$V^F8La5sbS5hj#$~m_4(l?=Ey}3n>#-|g@Iy&du!)|f0hx@RAMDq3U0^^`kQEBG0K&6DbTST)wXEzV~v)I@^c;JKII(}I@D9Kryo+^3156WqXaKSh!c0H4#-(oz=JlLH3V!pc^nMVcBNv&)K9{@OA5uDAuy9jZk4Y^J9Y+quw0rSKAj%j+JCMbC3J?o7iqS6T-yP@HJs1LOlrJEqx$6-_dWZ;fc<@}UZKBSjuA(4g}5jsH4c%7WejjI(pl(k-&!X9ym+VN<)h=uyJO4;c=I#~9WrR~aRx*0f5l~cr&$-(EfJ6LD+4^e>@BMm;ZpAuau%F5j%ei%_&<#{r9CzX3eutKD5t+1g54nh4|Y7D9`YA^J|y?8AGM}Y$nPggyYy<i2$Czhp?f;SqsD~o+IBpV`7nykXvdir~&YPh+;bWGi@BU_IjWHL5Fj?yN})|uJMpA4DosdgO#t48xl^X@uDo4S$qxKW;C5gaZ6a_8l8G*TNQ3|yyRijvrJ$*nh6Ab?f%1^`CqLf=C7#yMHIPH8Gr@Ana%+HY2gf+w5stqRP0Ezm7Zm}2F#EYd}bAz(O4I4Aom&WhU<0U1Z?e2Lf`GsV+O_M7073&DoD>GV#dj)Gzw-c33vMWk>}%7NN~Wq@u)NSB_nmfg+O&Ns6kds|Tr9cB!xa;)MIt?`MJNsK?`>^(k)Q*Iie#)<GY>BB{uXA!e~*ED&?rtG6@0?iy)1y&IA<127va7U+g0V9Q&Xi`tF4dc(9w}@KZ4vc&-p95kA2T4z};|ak}dli%;GvMBrGZvM>Js<8V>h{{4<q};=+|QJxNXe^UG5+1pSe)}8kmgU&+`UN5v0f|R*L7LIUhwjZg*s4W3WR$K6gm%b)2^*$=UYpL;-mBFE-LltI?rJbBO3c?uyNptBBob`&?QlnhSd2L{Q}IFE%%q5iwVgx2CeOInD(@mYWUi`J?gDD;iMkZopo6)%I3CCnf%GW1>nAhHDwHs-GwT;Gc5kDKIWzb68omeVp2n!#bv90IqrOLFQ&Ai)n*jKQOJRgFDq}cxnj(>Qm=&utn9jya>7<bxpnb1)E$%sEn!W{lEhDkE-m*axt*JKJbdRGpC)Gf=+hL%j?&5ubX^;mc$ABV#j#oz>y>M8CcVxXWlnm4w*uLKW$Im<-sd%mNATKHM;iJA?Q>vlVkGYtVzTdkA#&#=MZIqQ7>w#{P0B;JXN7ufg^zneuQZ=%mq`3%Mp3pJ6+>N?G`?!63j55XlkFAX<#UQ`_5qYpNBZg91AH<V20jwZt~r+8?inzOWrsbB-pwjo9>-69$A2$5;?XBswZ5<j^O~ERX4e4oN)5?ar5!ynxb0<NV)bz(5U03ixD8yVIjJIrIdgk>`g8X|uzG&0;)B#t=?+dn<Ax5$j$xK!AY7k<Md4xx_6u#5TDvCGX*vL$RdeOQ;-lD2vMy8a*=5h5@83o-_D0kf-Kg;d_{j#3x!Aj1&d58xsL&0R3>i_Gc)vg_4r)rf<)p{waFFX9{M9%;%elLP#GDmgvofS4Dwc+MAJ&&AU1N<~3i#NPLaSw1EcI8`!07|wsHEML@|t}AhBn1(+1A=ea?`pUiUo+y?{ZU5H|t`}xaLNtta^=<)4K6LZ)PDQS-FcnmNa<L1=Za$I0rmq3d<?AYrtgD_WFyLgQfsJk&6_TZd-z04V0xkTBs~$*H~bYhz47GGwx!E)CP0x1Bpy$(jHY73EQHwXR==rALdux#q7p)&)z}bT6&NN=h;1u0o~P<hV2;k#kfr<<wHU`(^(R)Fe~0Si^$?LD>eynON6q9c#*|=>nVX<_FNPh!D#VMN_C3fjHBUsIh-0JV%lbY?X*(R$J9hhL5hC<c1mwWCxbeJm(0;+7Tn#t&pbx{lR_$q*Hzl-mCRa7KNHx5Oz39}!x>Fht|)AQO%22=d>|}!5*=-GcN?I*irO6OTAsiv@f<wTCong5s=R92u_76Ne;`OwK>aF_6nu@BMmNbE0yK(4FUbD-Zy8a5uQ$swiAT>BW(^mCs849@erUxm@XHcOG*t)8)HBwYpHKN}Ks>lWE)Aho2sR<AvM1}uq}x3fG4Q9BS%n40x|pxYa{PEXzw(mdjey?8h9xZdWX<`AR%dc?&fDl;*qb%dn3Lz?O<1)uB^yS-Js<48A?J9Qo$Lw6A3oo*CArMui9KSJ@m7|IU_(Eb8VMG}VGe5-DJY`$@`y=!OES+*&rh59ru+6!lsZ{PR;4VzR?Vwz#@oSsw(ta&iY+gUyeq*^f>uxpRqbPMylHQNyRyYm#e|d6<|a+LZgdEY<1Wfx^;gl&Ht!zNC>o5jU?zVossKHb!ds_@idiu{p<wxF7us8_DDl;kmSZ3317cY&0kN6V3m|Z}B{Y~r89TPX5}GnU7c0>+m=pW#p-_tB8_y@%ynJV&FScm0j{tx-+25^$5p89EIl(MEyLHvoS7lyq{h*?{O~{A;3Qf~<H>E};vmjoeAtG$(nJZnHIF1{>t3Iixj<%I+_Rd8T=yUT2<VCOZ`{MWRb>xqDYFZDMv$8AqlolD;Q?_2;p}q3)H8G)}A!$KhMVhI@WO`d;YP{q8@#f1uLOo{1vS>UndJUyxT1;VMu6b0>eP>iLT_?9jQCeT+E>Pfe1E6NMv=&9lmxQU}hg@eT1eje2Je<avPG(`nYOUOU?Ku8<W6wLanMPu!o<mXqLNu7;xYy9)4R9rI@r^<vroW|O1qdkH(PrUblRPA1e=kmn3hd+S|F<^xnk!Y|5t|}4*#^W^k~^uw!Xl<=5qTrf9OFuTH4IxH<pF7t>h?;r%b!U90~ic39WNGrKYFh(0spI;yryfrRzl7<Zn6=m2^bC!k<Th}^!~`MXV)QQoLQ2o(3lbk?`g(Uuu(SkIaPS-DNyO^%Q_wcCNj)9auv!Kht&ElK#Zds7=liTj`ie$Ze+mxm{{0&gg7!Wr#f=QM0Npu(-P}b2Geyn9HE=?I4?cnD|n#`#*-HUyXic=>YO$YcFO`dwo(T(eE14C$eGWIe~y6|qld&`oL^jTxObuK@u2qr$(GG0olh1I7H>J?EPGDCGlI?z-NG6Yu+pC!WT5v*(^?L=n_I&@V!q%9L>=nGS62+lkSg<UsXkqlSCipyEGZfesu-js%^Ob?FsM1SDGxG!hU+CQ*&9rA-T~6M50PQ+bFLSOIn$<1DePs~%=Q@&3TT6e#y15Fh_w7Pf~0$&7sPgMfRrqED_<=-?rR~{{ZEE$4E`aXB=vgCApGo%SJYzFQ}k3!Ii|-Ny`SBD%VC+kn^t%2e5hxbFCN#6aG}Qy!&19A+hRpA9C+_#xH_!~g&%r>Sxt8(W~bia`?tkJQOSm`xZQLbk-!zu%)W>_SKz>Uf_0<lAP*c{$%oJe*4;*u+R7VA?wTL&{9H^lI7%}Ckd9(rpZ8UiIUBLy(8wCrRIfPtxc$lv8N5=-)0$2W9@b-dahIAF1Vv}zzTb}MC8D>93jY9DfZ=06)+BN_7SjHljz0Dx&9brAo#kFL5OLbU7Ex=kKwjzVBgZ)YniM;K4PJ9hHW12!XH|;b^mx|ciEM;hy{I;xcB*C@WD>OwMgtDy^><$!nHZj&pbG#7?!lE{^<!n$f$3E!m7!xZSH%_c4>Z{3D~|6fJcMAQLGFc^Nj))QNNU63*(_`p&Zfd-)vK0=T=@WELsv~BU}%-*k0%tGLoiTRFPM`qRK>>{&bZ04@Emy>J*o`!C;848@GvLMtkqoOL%V>y0D6Np%p&-^zQ%DmRk0;?MNUv%M^F1J(e;_YliFY4Qko?+f=!mWvA8W~K_}r1#4#Sr1ZXdHo^b`$G^b;auRS*27=q%%tF&y|yxR{f>@D)MxM7yJbA(4e?ci^iTu&AD6`Q8w=BW@^U76BbxN)9)AC!Uf$#~}pmrKU-K94FAj7DAk9d(CAmhT8A@>qGo2Dy4TzJbpN5?iniQzz<`k9-lKmP>^+u`ycjx|+K6hld0{NyI`5f7o`aclygSkK}3}+j}o$66()Hj(M~^wIj+&a7pd>ijqJTxE00*qR;5Io}6VC#-r${XABS8g%S#F!BR#X7yecPJQ>2-OnsIZ5k}*wZvSi&kJg3uG^X3)^@2VN)>J6&mC3SI355-U;HzO+jx~$B>t3qSRMDXahjXK<i|9#ZX_O9P<O@zgLVe2ToU}~}_4*kOm)|F(UQ?2jAo&9aY`X`VU{GEvfav(fy;D{KMUn)WAnU3@|2%D3$JJ9u_YOMysloEZd3OgP0u!1MEk~`SDe+v_<=A3PPX@{ttXNPX4>D6*pGz&{z`B_Q`BvY<OJVOkf5lnZ=<&G7sQJ+?P<hk3U+`}gh_?OHC^J{w6kQ^?Ge<?q8M#AyOL`iUO7cWn7YfPO1H>HZ(iwBQf-89)W<$yjL90hQ@p{IM*~5JM)1n~d@tSS;xcLWp*EP*?ME+qqPiBDo)DbF(liq&DzuW_Cd)!h763rZ<%p|O7v`_STRtAk=H@N^BHhO>}TxLjiN*m#k`dl}%gr2Y0vL8AK4N+{XH?;2fqF+q71&E}YZ+`S+B^LpFnlM5O$J_4b`!SfZNWvRRv!XezU980u;1~65`Mjs<RaSi=*v1V{Uv%RM1UlUgTin3=mE4Vkbax1)=XJW|lJ94Jwntl!ycLxvq|~Mp!&|NHM947K8d;dE#7CUjPZ%I4A>b7XU~{*AN-91%pr&#r@!`@GIwC*gI(_}{2aF~Yc6#F?4t7;M31;~`Iz%3??z2VbrF$OEG?AufLc<XFTv0_pycN7ZB;>+<+?*M#D3q=!+OFVGs<wPPScbxXsD>Awv~3N(9Z>|qk6qu?Pc|q>ry!V5hnnULDRX}m%z672It&T5CLO1dRw=Z`0mOkaeXLAHEHW>ATOd3~@y2}5sm$&yyrO`+`=d&icPhcIH&OKrgPD24AS_DIEO|zN$yR*}-kc>rU3L(Ks|C2tDQzH{DZItEFag4xCHGRwB(2RGC1?4dhaBl47#s<^Fq|XJ)6ZJPIa(salQ<z9<D4d4lPio<O!z88T!TM63{!q;@Gt(etj(ZRv>?;6IGw$VzI&~}93K!LpJ=;Ma;PUw^pq(tMyQ~fc`EeLP6MmPfxB9fW+TB<-Vc50CaWS64z0(`{<fw$33|}t5E2IDQGc5m=c^AT!aCZwMP^{S=LB>fJ$y`4mMoX_#2vtw(ISH;DBm=54opOuu}%@bIFq6KY|_GERt~sQsZY-u!b;nL2g(tP#dsNm$l;>7FiY->OWv6EkiDjGO8Owwt!HcJtrQq<8WX`i@|qo6b9GMv$$8rB)9}~)L-LBZU`*3IhAa$>>=}yC^0uv~>NIIce@x15gb>#1Gv@Bj^MfVIi)vFh4_aaQ=;mj<dGqL#dRSmZODd~Pt}PWs;$;n=nP*5_21tz=lV=tPYKo**x0sc@IdRiaeWy*e@gyvMoNmVs+ZjEYv~@)X+aA=yS#hslH#}IMo|xuIyy_G)ZepjG51oRv7K=I&gDbE&!?idMH-c#T^Qg5p9&snS8Hdv#k9WC_pBQCV_Ni|#b<~5QHq3blyHV|AC`i+r6M0n);MbTGlf=TiO@iHW&PvT0B0mKmQX|&6vv4RKz4Y>WdFMs5+K5MIULYnh{V4208?O|NIMH%L@dPee{Tv@HSh#Gf-YwI<gVhBwV~SO5-{yDpv|8NBw7WWA?8AClk^PMJs5nLZiK3TX2i?swnI!_YQc>PBy|^wM48hDUa>!&p*2Bw9A)cC1Ac*%KU&!;I(86VCzMAFAtpjC|Hu(Rg=q$Dz1cE5~K`d~|68GS`gW&G=^~0=3kgj@lZ%#wV=lD_#oZsh8Ds|Agl*eUhqFeiGVR+bVBypD8$K0(z*iYADW<2M#4Wg1jsOp=X@yKC^LSOZofL$mproH9gJCHn-^`!R2s?=^CcoB#hMNWs?cVA<GtDrtKB7lwy`2fPt-R5!@=7m+gHt{^&<UZ9UF>|n2q*|ze)10~G$q`Rwj(7e)F1ZF(|C+-Fm}DC7N2C`gW4M6;Ql3Gd_u-JL&1d+Ai^X`w7!xgghCH`Vfc-=hr<K{3kFi(~jr^A&)g}%LGOm$K!`54k-=ui%B2!S^q$cNSy~l{lul8deqD1!}JmsB4x|jtfaxOHQla+2t@wN2w|Nq7(C{#hX-&|7SSJhb|oMQL$%YN8xwb+|EiHP&CR1gBEn))E1msNt}Pp;;zf<$~VWQf8k`N-ytxD!?weSUR;S~(rVvE?$D{>u_^bq@CN1s5>PPROj=g!w^lBTd~1>|-jgmyI9do)GAO+yPI^`m;0jLY!DR;r>)`gndJcMxLAWX_`K%)uk0BDFC{(d0vOjV<-m&Q(%fB0-PUa;AQ<{w62TWc{a#TQ3*P<UGW&nr~+BV`I?Z^%p1fFthM)^Zp=j2>vP!`*p%#*Qy4GC#?hjD@o$jAZkcxnti<6)`oB~k9brD_OtpYu?)m5dwnhfK&}|Q)y~BD16s3;}<IYL)bJ0w65LH@Y$g&Qot4~JuPO}n=>xY`vGRjaOV^%~ZFlraM=C=ZB7tdD-n*|OnZ<~*2jxw)-h8hdQjX-X6ei!RKUrL1JO$~Dmft@BwKU<c4>*-G9Q$f%4r)Js{9BG}y;8?~Fzq1@_#DMD|;?I<PEVP^Vbbo_>!=R9mQNF|T;2-HeF*qcH;I0?Ehc@5b!H@G8hZEz$CGg2NOTR3Ew-ccfK*L}kO;@I-k`WZO7`Li+X}ZO>_eVgIgKB^^?UBv;h8Lyj&~#1w;+ZMMo%*Cll&-A>GphD^KCN54XmR>k+Dm+Aye|__G=7cI-wnR=6@&bHm6E4OyV0|LsGx&YpL{tgsDnX*Sv78up*8V^w^i4dB4X{1WPK~{A)BlZk~OOwjma8@wnN8eP85gtc{5ThQO@KR&W^^<9>$}6JmiZg5{wSR5P<5#P%5gIUn^K)&H%7UYVI4sR-BZ7TWT(#eSOuJ@fzm&tZ3aG4fCy@WRfMS7yuJuW&L?dX3BJXXSTD*9ZvDsvPRbz_I*|_l@vPZaVRP^LB|rEmebkDU#dbxnyL%qC!T548^M3Ar!r>EDpnN3%Ot_ihQ<6KQs}fPFS#)9xHN~Klk8|q;nl`}b`#6pI4{4wf&Z$;Tt|g5yFdyL`;t>uMfCT!fJ`hedzs65Vz?We3Y{+DKbg9SD0Oj)u!8h=2UlgdJ|hM<wsz!);9Akz%pZ|k5{FjtSf%P2+)m0d2$(0SMe*xf>>Um#2(Wp5B^n%At;t|?Os55KPhY+@1cT`0Lu+5<=engmEn(r*I(AKdMwi+ncYl8$(IDoqV77LXuMZjBG~F8hEz}fDhtJ3ye)f5!IO?Yyeqor<&ueVK22Y=XE?Yc-`CK*<3n>=Nw3cLD=1;*X_6;NtES2P(BG!s#@`P`ZI(j>t*OSv<_+F!2$Z3HBGY@|E?@;?>{HNo-&ZlLj#TSa^6wmCj%iPPtU;dkaE2B-!IZF!p)<pA+DWXB#Oo-kr-Ts@+F4@uzDCeZI6=tluq*$;jE4PUju7xozu{C0GI2AZ%lf=u(0JK@aua5vFIocX(4Ey_GBoX=&45roIL}=2g;x~c}mNiHnu1N&mj|6vHj4r}2WhEW`n(n10bI+PT^&xe+5VrSk1b3bE(&m`zcI^rt{x|-KmhLW~v4q$y5D~NkG*eB6*OIz!NPfXP)E5Wc;j-d$?A+OtN~2kpF2t=E_%;fL^L=%iRdw9r*Nng!%R<>LMlS!z4o15rXb`E(zvx#(mdjy3mW+zCPR<#OY@$4xw&HW~{3bGqqkgVgJq!!*c!a9M?ymKiqs)5>u0>v{ij<(^F<`<eX+mqd{*bKw!I;TS?PmNitRzJo`^l*WL@MK)vv7NG1`SvCMFpZthO6J4e6@#bymu3!NPQlfm=ZcwT?iHgJ-xy_OTpccG2@qdz8ZbSu_brTGGoAOe)rII&DyhelHl{H(lNAKa;;2OkFCHy{yzKyC-b<5jKRj~dk*tmu?ccvmeip}=Z%6J6mbMLzaQ*slH{#jfLZz35MI6mrdaajpbH>ZCt^2!=!3rR_CDE{VA{!e@thy^j)N}dD^sXmI}4)qKC%y*wv6~>=~ZAPZgpMnaK`M{#c!vVNU1KeZ^G@NoEF<!NngZkuI-cSupU34+;~s2d|h4}uqxmU{(Q`v8Q<*d%5~4|p*IVarLVc5%Yc*)NJ?9CPluH|j5@$8B4YQsrGKCBYGHtVXBR<)^HCybd;SibAgF!Uo~)gHc|1n`mn;e&zk--hxY;zoJ{Gcd9n1`V^`f+b2!G&wRpF;H|91xg`SW&Ya;t6%PF^$>S&uEr(~}DOQqN(~>I4lxW;7P6B8$MB*&AOXS#vBPgW<yiWA*vc&Wm17P8j&fuz-SD7N>=&?aY#O^Bk#u*hsNngy1}r!+?UqjUL?mkss>;dxu)@)04>EYTMxW^BPl8{ANiM+V%NSV2nJ8(~OFEv<#M)mS&6yQST(*!Gn+Vxm*c|Io^ThKf)QIjLfm;+S3Z)6kBUQYz$Z|Q9P6smWC36&Vl443S6`Gkq>m%_5JP7F(8tO=3C@$vNUwHaRc?)920qi^ChO@cbF{)D6!gLH7DlSKcu3=n4QK5UAdoT2Yb_4)k=TeDhbT4EZ3^Z`S2>}yysw+2Q4K=k^3h)4Gjvrqi}1+LAixnVoiAO%uAGh29gUL;KaIOXyf-Z%6&<2eomz)Tm-+ar3{SD@p-FSoLs@tHK7<N<Lmcs_Q~&mbAA#HH<oLLV{L$YS8*(6s2nuI?fHdV$^5+(HPvr_W476`R=cW}1#zT9R1OBw?_5Eo^%VT;217Q%GjkY)rn%C-T~ZPHEgx@z$LSL`v8LUy4Ix|1d7`~;gI>4X>b8$IZ@~5=%irNG$xa0JYW%t}h1NbA6{*^xJe_d{9o|X~<Jn2T`$4h@xR!4V$SOX;X8d!{)`g88jdss9*ZJDg%Q}h>p1EQr14`~s#4%dzl^N1tpMZfxN7SPsMTN!Y7P}>jj_{BR+}QvqB&Wee8~fS%RdWU6IVj0^cY?IQviN)LwRt(L%4$2vBzt0H+zwK{1m5=1aM;-A;!g2@qyMIK{plb6f1n1hGCZQ3EyJ3Qd%L-L3U4jGtoAJy(XuzUM#*cEe;Gf=3~iUWnm1%y{FvTRN-9hY1;>V%a~>8q!SPFA<jXQg(4|-^WkG#eZMM3x!AvoqbZuiCSF-wFqF95+w21Ik7F=M+*!}eLmRVhnfYQwTIDwBSkWSqMh5?_JbJMD^LDEXpE7=B2H{}=EBIapdfzYSAn16R$>n-nnO84iciy__6NOeu*2ii4{Zlr=SM$O+r90tWiPEWXFKBrlC9S9RX2KIuoA1+(iJ)gk?H9L;YgY><hc1;3Vod8iu6cs4V9W%Sf25Gc*w~D7|NU$!*+wETuICcDe9sXV2>W*t&VlVa*C}ag~$%|QzyBE~iD!ZMD0?lsj!9ixi&b^APjNG`dOJUG&8l)nvK8ElV229c!N`WmhlWj8=&hK!-M~I7UUo8=t2X#CpYV=>pgk}_+^aQf&C7*$s)D%`siB}KNEIA*I{u-=+rA#l}ZwT<X@;az=lfPfySfp2;2!vKAN9Q_i`Ch4AQ*&<>y?`C$+WQA}y&K2B4)($a7cP*(^1M{T*0q8%)O`I7MQV#T+FKd`xJZ_w1n<iWXpEiR(^TU&#t6U5Sl?!MMK98IgtP*4Fm~aqRXSW+(DxtzLrvUch(aTTW00uh>w?RCeNEmzI=0G;V@ra)4%GCfjSugtk5~xfeS>KvU%pU*Wm)<fkABpQQumkRr(5`>D82~f$Jic5sR$$qGz0S<)&Ba^c5L^DY8?V_=&rRL7dV!t<Y(ryVu`*f^{TB(rH=Z_I`M~GFh-2+G+6jAYj;4gBOivrweN%C6;!$`iG^2}`s{22?rDuTu4qwd)iAhoNffGE8X*K`zR&@@M<iv^cD${4`Kfio9)05U=xZFzu5%^Y77=d9N$|w~RgWP(Ue#n^Y{g!cP#aM!AJh3nja{}`4XK7|yWL?AL7y@g%7j`LOLkAa9P-vYKZ{pGGyS=(@w5~Jf0^~#g3_MB4iQ`H>5Y{ObxJ}{+t&mInVXC8@-?htJ6--V!+tJ2f7cUlbmGPqqZ%^Yql!x5=xjbrv_mdBiWen`Rs5yaBNf;`FOrq4YwWmD8>}XXQ+`RP0Dsxkj4F)BX$hP~3U~Qx++Nf}M_5LX02=Txp1U4Cj^V3gDT}uuvN}H^&}(XOtHs$E#sjFMml-YtCIZd{97CCkO!tJTT=+a)=*TOW=EX)K1&MAJB&tZ%Q9c>K!qn4}MbTHCZGR*P6ia=aQHx&7BlwG1bE<uOoWDR^Ld7qiM*EfQ_Zt1bDl()O8U~CId%!*;?+zgjl!)fRi_^{dOleC&ds71Z3rC3Oi(N<iZ0OuEsDIZ{a=*7itTEM%mpg$c_l3sRPwl$5#;k=l?;**<5QZDE4tei;t2^9T4&{?QF{Z>^Xyf3Y4vWrH)dYrZMcemtmiCmQw2Mf7E@(~CrCj#qv*^<tOZ#Le_F_-4TdbqZxayV8L?bsx2D$9lL-;dFt=;2#q*HHDa>q-Zh__w}tz{ZREVnu$qee>oseoa;tsc)=`FLRv+d4X;kLDzs-a0UUc6MPN=041L#FGND{Kpm?QuH^ZtW|!Pb>(&&ret(|;;?b_C%f1=_<LU&ok?%-=0K36MSRywM}KvyI-I(=KISxp1?t7%K&bCZeQIRcf}6PHov9mdOqb5{O%xM8g8YtzGDj{|P-^^Ev``?<{qbE7>@T2S?7n|(fcSaaF}ueP`(4@CZ1Q5IV;+rhXk5}WrFRVAw!{0gFb$m73)0ZBlut3lf($CEe_RMUd_OIeYvg-a-tTTN&6>=2=%<7p2H}pbhMRx|HMV%CI&Lutu)F!&+@$>85@A_HRz&{G4E@A`g*<gJZiBLb^f51iQoVuLZ#xe_bmQW9Y#GL4P1<XMGQbl<LA(^>K+)m=Yvbefkd0%UA7LXvGGy(@PQT>pK;!esUgNz`1A7K>eoHYAq+H`&%csky<>|XTtV8h`s}tK1PN=k7K(82P8j}Z%`FU=yVL8R}&b@iv?So|6@mbvNqhy%2(%5;f9&!;eSFbqTu-CSWRlgNisL>5^jsB3aJ<oh2k7=k;+nS!nq|I2E|CGq8ypNeR_P51c&`UAOIO^gZ`LfX@pdVo*inMzvB{T9FHG1mc@q=B?b6fR*kf>V|0M4GD<fBM)vz}`jUZoB1Q{jmp{L=LvBK7p{hj1|fr$T&f;L$TWGl}4?@E6Hy^(DiwqqM##_B1ik?gLw-Pvx~)N^TfEx%%_&!j-eQ&T#8fcVHu|@kvaMmN^1aW159aq-P9Y`P%ubN66EEs7idEe0o#(3uAW-0bo^u@3=kdS<tnfw;|^^xP4h?!=|T{@O=ZdacgRA&-6|BW}z7GfXpC=V%MGCxbor~EnmO$;&n+dakWvc+ZlKzM2xC>4^cM()bJ--Bhg*MHX}6l1dNT>On*^4hjRVRV%JzaJUdjlCEO5s#i(=*XvPZEo2HF-YJZ@IUwxA~G5mW%D|}MmA+=8)xu36qRTA*d9NtdX-botR$aEdBIj+b~{ZTFTI-!DS=5L0oK4$JO`|}nVr+Eg3fV&af3?3CC#|^!(h`|0rBR8;oY2SZA`V%NYuA(l81lCUQvE864k*M>F&E`27ti8sEx9@jG^Sc|7aJ}DkKvcMcKbaHnB3h^4JwTF5{^AG9YIjH4iF$rY?z(hHy2v>@8R$(-gt{jcwiRK^pA<0PE_~(BjdB-z>NzMJbR{uc;w`rw1!zwMag6r}D!v8z7QpCSclqt$h3}1pcI}XqalyRM2>dAY=eMQ&DV@4Vo?BcZ?`S;x<kx0rsKnF_^ZR{-03DbSw@9CKHb0#Y08Gzq3QJ)Ess*_7p^?H&X8f{U#FHG4(Yk7~XC!as<b)%rj>0s?WE2a_pGEMI1s$KFfm|r+oTd&rg4;{YPuXL8N^U9M_&j4JE5@R!y=lZ=wRY|KsK#W-pI;q}fPO=}j+%;c+=+T6m6k}R$2f~RvsC@vE<**gKuF!=;m8GKn>&UFwQRN?<RqWKafAh9U6r+YtN>XkKwc=P#_*eiObmShSKd|}ZtUh*#!4^1^B~+c#yOQLK6t^)^TcDSqDE05P_BXeuEn=Dol)qFvvKb)>_`M)EO;7k?31dzrS|}UlKj41xoC_BZKx2P0HwpJ{Kv9R@DUj=7jIXeV$X{L8@iPZAiGa4rx=Nd6Tx4WUlQb#tYnmE``!*ts=>j%j19LyfqE8sc7o^ZGQJ-uZG8X2-n#CMO3R&2&>sLb&!@23ZF%8PeJT8!T<}QLI8;$Tvw!x+rnH9Kgt2_aZz7(3i5)zKn$g^Fv6M9rI-!r`RR+dVIetcu>O~1}oUSUz(IAnvFha0!NC;B|MVcCSbNqPVw~{D`eKkIgs4W2jA~&Z`UIjnMip?O($Uz$QG5clWnU*VN|1d_d<8=}USQpce(m7@t(!*o=(yP))9Vpw44fpuC#8IijB2@6z{f5@bJ=8b}^LD9e@Np}CJNsHHIsC955qVK}_GfsiS+!M*k#nG!0Vd)lM62%K1GZjPRb2%@CAOCQAL*XU|LzwHVGFVxPY&Hpks4$QGaFR7Jm3V<?0P`h9j#ui!D_%CkIKh3tSNe@XB1$8pb|)@1mm8t9^QI>hZG73LZhGZGfP62Q?V0D4C8*8t}DNU`Q?CItM=FLZYkv43w?s8U^~M^va*WeA!5``HK|LWvum;**Lt1^?SiRwd$!~Nh;caZcw4V%wYeH*CUlNVEhzC0j)`Z>%Ir*C9+8r$nhU<NURf>g|BaOLaFIs=`=uN0@C^T+xpQqhZ0v50KfESJO3ViSu#N@$^JDSrm6Su21HapMQWL2EsrNC_vx^Pm{uqbF5$=^8KZpB}xyaa{#p^8M3|b4k1UPTcnMyrO<45@aDpjVj<UBeCS9LJL90o9>S+vx|<3+cDy=@Xk;bbvw*&{s6PhCqsjX1*O1-mwh6oxPe{qrl;RKgAsaDtK^&uXUiuN>6rr`mN=!DT|(xRDKnh(MJ(QKT<U^>wp7MT14tMIWN{e6dTAvNY`RFtYAZ`l7R(`n-&pFHe582}Tw$Dv&y@L4mYB&wg~-cgJ6znbaR~xVXw<(GWD0uIzBnp*Du(NWaEKWvAvhpo+WzDO&g$1X?b#6;bG5LLLX%OwalWL|k*X%DfbAk>Yww3ya^j?Z~-QulRvyLgkqbw{LS;`1gnRMGg1nI(l#Q<Bd39nSpAic5>I?BoKvt6jZ<hohfzzindi=*p_B_9|}>k(imxRzzgOzl^{KDcRigS>Fbv2i^?1KG+H-EycS!CfgO`#kHML~_DIq!9Z_CLAtWr`H47d*18o-C>ij3FwE9zX9NC_%moAKxka<c^4Igk%Ejt+6>x9|jZ@isfcfM3m!TeQx6CdL>kI;EEiLk5f^&}99GuHG+4jCUXx*<Mt<fAa1O(biu?qPHUspK9x)?h3>&YY~c+5M##M7Pq3l+~!G@ViAp(+jBKrRyt{`XI;yD*T4_2KU1HIiv<Ze1dlG)D*FnP<Gv1Sh687;@6^;O|6ASe&7O^@Dg{BOpRo{-<BvaGfs?#@Inj$?|8JFzTol5ng#zjw892fZW3gegB|>~&bM@y@N<7>qwzNawK${NmVA5!H2GnlKHy%gLU~Gd84n}9OLtgYE#Z&-i*>u59)y)dT{wk8baCJVFEwWVxO!3BuT*5uH#)7V%mv-7#p>W!cBX3gqiFrE8f#J7lpXUvTKK;=_Vyd9v9Wn?_o;t&SbNhFb97I(Lde8_muvwW?-tkcVbFy8_HTDUIt_!x^w8_T0w61of+lOi1Ejb6D0?IhDe;c7RJlw?Bj{$OfGMbdlCO((*M#o)<k_O{)i!dedm|==Bnoz`Wq^Fq570j{3+6`p`e=Dx0Hx$5^jz%!6T)-QdKbBmJJW1SFvSNYogGVS#454n_>wt|2fPpZ>99=eDc)sOrNT0a@@iQEps-><c>}QM0dv4k*kf^s#;$L=nZUp?WLctqN>qGwy=NK#T_vWrtc<O3v>0$96#$*5au%;(4ROGNOLlo|gl{db+_%bP4FGC4XXtNUdi7Q;9e*P<2tO-Y30HO~pRF-#Iw)*2Kf3=<`SS}dFzYSCUvs-5_>&{Q=F_M6Uf<i?Y3z2z@7`V=ALq$L%o7EbIbRqBv59KTYQxtaI9K1g^Ejn$^Sdv3{y@enA->AW8n*=du6&=UD&~N2a~QQKm8bdxU5MbAtV%mkuF7!owqQs7+QKRjk=*+5R1CRHg46+w%lFLg(-wlx1cOFkn+U?fZ(*Ju^498|_a>DYDfZw}v!C3deDTWEoyct=W_P1cfnZLZz<{s$DovEnI+m=mB{?n$DKoXHlfTPd<@9}i`iY<&j}>`n;wAL^zwPQLK9s}_NRzx5UDDX#qtGjg!VUQ@W*l(#|I8n*8r#s5&Q<#|ze=M&tMa;<&Wb%=EJM^#Y+BqpCnB>sde6CqUyMu_Yro{(6hd~O*|8P+(swAZFOaVKgY*<+NY3+Np=8d37MSgGHqWiqQ|yc7tc-cd?_<Lg30trE4hsB<O|8!L5}3t+e~`m4LY<nGZ|ozip5{R6BIfU#J>h5_8#qHZe<ev0K0ORWge$}6*ROO0lx|?WvkqG2<Y(hPNxu~d{A!q)DY^BKVdP=*xg|iyewDOVoAa@Uq}t*CvxJH<#yQ%$pL#;X_&6UO97{e&5z7NmZg+f{>O-vT##V>eh52$i_L_(1hTreNwQo|O1^d9}Jvt$hOl@>l{0)<nF~b&%A2r_1kZr^K##8M%8*OxlQ<tS2#%5*~X2CI@kXmPyP5j6NtEJJ*q+=6mmdL(+6aMzp^Cn)`z(!FVXX-6TuuoN{icEz**ji)o!%bvlET{@j_GG0Q2W4wzsb?&Q)Db`BK%y^t`I&L-G2hlAJ@XEG%p4u#I#QNf18IW{!<gm4`R<FHQ2riHi}AwC9>S!cQ)?G#@xs0E1!vYusj1PRQns+?aa1d<rggbH=*;obznA04ASR-1vm`i{d{cf$H^Pq<-ZKK2MANco=!9|#&H?hLT(Gp_rynUe=+8}wqfA+J<0kJ@2{s{o4CV@ks=ctZ(^s5zBcQRJp<j@1K>O?VV=>Bl8+c#LqU^;j5%mp(3_#w_%OxFZzXXfYV{r9$^QGV(Ig}zQ_es|-rLt=(gY<64arq>PjVsFTtcR?8B+Aw^ZmrqoBe0E)e)|CsBC6Ak1RH1RXaNalAzcJbjaNe<b8OFPV7jQCt1sJnhw>Y~oFvYhBQEQkNc-j&wa=%a3;l-?pS3DX*>`DNJ--Mb*l9N|@8N5qzu65kpt4g-C4+n~;yz9W(X~V?*s(G<z1bSi@~M6q9spEsF34y3QJf4mngYo==EoJgYaJ0f1urdYxFQ5KVomK3aKdGk+%enn&8@O(@y(@s0}S(YV|ZMi|87N5+Rq^@JvtT+KMlZjPM;kN8el$+I|GP-ouPeG9;S;uF8(&6&zt*UY*7|+!hx6f0=JIT)=*;S?W^xz3S`76Hl-8u3dqA4=7^z4*yng?LJa!jb9#pS(Pp)jb1fYAvY6zud@bZqg=Sz+$?OL6{A54qQ$In8hzC1c2_3l~lsh@?+hH9Y+kR5^A8~f3NTunt-EXD^h?2DU5EkY7y5V3)%i&7>CErEaAtg!%>xn`E$4jLkVi_K|YH;H(mK?88o#z^g8`ur{D#XMT%Qh>}dtYv1@c5p|@Pss8XEJxzmwUVE!sFs)SzzonWV><B7}V|*$yt;!r)~7uws{I~99LEFSLEO3br`2yhv_F$n__<DgT*|<N-KkoyO)Z=`UQ-l-4WiDo6pK2q?I#JZd{I#>k?f)qi99!xPK8ElMZRj@udVNrM)~Bkd)o|y(3H~Pq5XrLmjW+Syg6WJI11h-Y12=t5U|c9TnG&@(7#AN2K6}y(HY|HzcK}X9iS{XcdybiUTI9G~ESIgeuZa%&b;^QhTZTwexY|8t&XoWt>y=4))}Xxe#$m?xNd1)`;3wzECloyuqg9@@6DJUoN{_(kd{~>Q>5oZO~BZ<;f^lYT??CVjisixN{p*#>^Yx<h|O&BtcUi$z6Xr!9=2V{GiaU91725m4(LmnDq|;&?oe&tR4JSwePH}<A7h}4OD*t^>#tcw@{1iKrFLJ$IOs;%H0R2qPxO;P#;U;%hO{jGRI1Ck_HO_dm=HxMIh-(gt>Q%ET>s;QVIt2)&LMHI`aMd?76($1LGQf343pqY>p%%LM&>&cj(0DN)K9%jJ>Y0)(6kdcxPLXi?$Pb?${evUxsnWmQ+x41eC5$;JbK3wVZ*JJ?VxE6aSgDKz8#ODbTwys@4B>pRcLFqG#b%Qlw%Q;v#ma`J|z2_)U?Ep^gUmCgMW2yoxZT%*}|b7MNr1u<jO6ntiO6x<jRnNUPB<bZrDaDRLmmK1gk2p2i(J_I8vEjoi8rH*a*Ar!KUJ3k_k-M+R394c9*Kj%fFWaz67&lO<*97yIPzhr)X6biZOxwPZL-wB%0qQ!$do`qQoo;iyRMzI)LkSv0JzMqc=zd8Qv%MJH}b&0%Lwuk2RXJ5s?ZbAY!4$l9e<{Yo)*%6Kef;V%iLYqr6R7YvGr7V`GCt{d0@g3-zoj|u=rgS<8<)@)5&mBR&|1D({uC5p!faLsEnaK0%77Ct<xjhy8CXrd_CPi`9U?J_}I!up_YZ@%^w1tK@3yTQmH^0V)ii#UavdfLq+*AiuJcmX=+(Ej5R>o^bcv1+w~o03NP$V4{RnD85D9KT>)SFx^YnH#8pH4|;BoISCBZpfe+r5ZBIfPO(j>lS?JpQ4uU0vImju&kVQpGYdgH7t|EtxeZBdBM+=q3#h>n@ZHUAn?~h1pLnRU%}FHD00Noh^ew?^(U<MaQTJoV-jn?4K`9GRQGi}e)ak(^s|rNl{{cQa@;@#oozg$P=dqW*sw04cR>~XUtPj=ils(;VYgW*jxWP)`^{J*a1-0=l{2X(^<1!5yn=7Af=K39HA>W7Z5_YxoK^d3Tyr{Ioa)IRq!tV4rk6#C10@O9!S`MMH&)}R(^d8Kuy%8Grp&FA_MBoRd_u=hyouD2sAT2QC9aESmy*xIf)65Q?6JIX)hWbFXf>X4Xjr;wqnFbimwt1<8lXKHNAOd7Fj&(#p{lVKChJpe0;h8;FL5im;<@O4J<J{6i_}l~XHXw~6}_ygJeU!W!cUG%)itsJRJhN?xo&vsl?42xl`cuN3O)DoDg^#{l*U{=h(9oDf-Ojbob&e^+SlLzUyOV#?frqM<c6N+xpQUn9v-2oW<y_1<ezba8luaD1%61;+`*i760IZ1<2O?!Z5cJ`GX#AF$?&Ctxg_7F@2zDtYAPHh=#zy_AZ<A`TOY5WAUcl#tcI*U8dR9l2{e0`M;z24h@8ex!?B!=$xjo`)JynB3$P<mLr2q6Qej|IX|K0_w^~rOqJe?J718g<qNrYqj=|J`cNX};<@ikH<RMwbvHT;h%=@syxQnMlb5$dkcvwfJ2rDfhNJ!$r^6*l^W~AA)ptWxHG_b<_+*xHsB5X9<`h~{f;e*A+&p(K*Jzgy<QL`HlB76`Iq<k#H@8^W-KEZXt>j&Y`APC?6kvnXgJQnS6mEoy8{L5z4XTPc1_*jTW(SbXj^h^8FMiZ7u1SVNwO|4z7sqTnPt<g8$d@H|6>=2llK1BEQ-3>)t6%uIqASQp=KRPn_7-q-r%9s$B!@@K0gt5lqdYm2~J+2MLz?PY10ewWjGkzk2Io`&(oK6uU7)Hu>{k{-#+ZDx>*3}9|={`?rHohZMz+MNID5hpE+UOqm{&o*zu9&hJ6gaTkH)nAFZzokEUx_#CGm}@S`!$`xsaPa6n<Cwf4iI0{faInA<0lO;jz+j&+BNlyJSj#()(X$Rw@UP;@?&4^(`r~@e&l>x+K#VJphPNQ{?-kd<M4EyQr@`M7ygnjn#y2r(7N{;RVuFz%1LSDArT-bltLY#cI1(P%kuN{%dt$vNH<YO!)6SZZBQeFYE7%Mu;=J1e~_e!b-Q^X@EytK$ifF9nuUuJv+`0!Wr#1XUiS5htreCz1d~h($LK^?<S<^`R{~@y*=<c#X#Kosl=AE6Kn_uW0gjFkq)_HP@_wk>1qcmTHXU!`(7Gc6Bh5X;ZHfZWHVrS;w8xKM@E>ybC9%mGF=k|;Xye}I0~Y&<)FfV=;juO{=}L}3VeSNL@VN7)Stp&KXw@3d_w7#m7OjaHBRb{$!pPMU>L{*R(l=0l+<QR&I^=H3*y3CZX!uV_5Vq+0I*0-Al!u?6hOc`UdO7DtByiZ;uLHm^u|JlW7A!8%{Et6SFGs-!@Vn8&wSBo5Se$ck9jJY+zscu&UaV}VCE7TxmHx~;T)iI-7!_Lm7@F(obF?t}R)pF}93IL~OUK7xYZB<g<r#VXaBC+i#s!SeXLWEy^K$URdwN#<K!eN&w>SFT5$9M@kdA;xf6qRo3^^|W<<8S}?~s)cePAHbw?6d8J9AJYphg{$2@D@NhsxLI=>L7!tW6f`&506*9f$(N>_%myyVYnRE`>OsU{UKC=B%IAg`(Lto;K)?ar**@`Ss*lXji(#0}6oL>`pyLqjE&jOIm<9nS@TGbbhD_cikp}9iNe~U;V&urD7j1o{RR`p(vfZV<XNfN2f~Ud>0-1^tjw9h=u(@zV8qbgSuxusV5WoOYdJvM_!CUWEKphw}Kr8*F7v6rCGJIg`ZhY)s*L0zwGTPKX6yIpC;d#A+K45?<tP!xle2_9&@F&Y9MpN`$0z(cskJR^QFZ{HS3B<-eC=z(n$9)v$PhWP@EOX^^2ClugE5X4vN8^7;X6_US^YPiVN*~JsDkqmh7-=A2d-?MCl(REpcsyUOMgVIKSAdlwZ>gN;OO_Y0<{Asaqh5Tg5y*p;}gnqNV&R^TpYq`89!cvW4FrAPDEyL5?Kwq$KIVlQ>%jd;*cYK7VyZv{gDARZHrfo&X|PVJUK=g%)S5oyhysn|=vcU%OA1q{OHA)5F#LcoE}lj*=FtmcAID)0Cg(V+7htVG5;Ysdh!@{`3H%HV!)1GPd*5E5lF9^cz!P&hoen6=C7+Q4b{3nnMc86p6+7B|3_e3DE)DIGgov?IV64);VX5y?~J{v_8p@-zfnZ#gaN%!t*NVu_kxpYWT&d$QfO65OeeX)(N!o*6dF`#*?bg!t=hjkxm=>)xQEz=5Nd`nuYnjZH}n`!?x0zVl4S<ac=!C2X;V)9%yDzfVz`0L4{Qmb^l`hYj<^B$=G)%f7hm{qSEtjJ#-{CbS53Aix4Z~ap1@G=<Q0F9_|tOQy`Gasohq_*|lEBPXLbx!fdidfpN+KIi2DHS*D~O7Pk4YUn5dk%!>A?lCSg{w;kwoBZ6|#^D`GD^ty05CH~<pRNu8_Aueu5(o~w1@SBdKtcY{3R5+@o!G4$FcYdES5h?8S{2lhwkANlaQ=_z}o=VQ{NFAPE;0pRw99NkZBRFR6pd^}BS6Ht@>2`DSDvgqpyiLNX=VQ5(+lJB?FluYIIggm{4vUZm!^!wi^(K=^Rc`ZW`RN0#eTt8z&*BUQ&5D6GA<@xGbh(vfWR~}MAVtRSS)&E`Ufwp|nyEL$y7)kmtTDc~yVkP#pwu0(G7UQX?n;@t?01k+4=8q|zry<-h4Q|T2u-YI#X4vpCQhdDc7SAFI|4oBdLwz)98R^m9jv)y&}fJB*NC%LLRZ;2ozdZDt7ASvK(PKBoj`m@fk;v>5Xk4pVaB6?`O8<Au23I>0$M!i%HIcYRn9@uA3P=+TO>WH{>tiyF@RPI9p4+Bwx#vAdw_q@>&1b$76*mD9#gD*$Ips<a2tYutV3HtDmWysqJ3eTKt9b7)@X00Q{av~>5eIbgdo^_!saZtkhO06xkKb2;53`Kz*s0qpx%dsdzRKNww0cjkHn1Y;-<@*MArfmUNU;FzOE|TE|&2NB-d&1R4r*4Guol5DD<luG2oe(!0Rf_k=OM_T&)BU5^ztjlBSc(Ay4pqy+Qi*JDGgl4+bQ*E#+X38(_v)_%+<n)j6Slq?rwx)KK&%{C$^E#m>l$AU_-gycHobW{w^!c=wa1dYH1~c)-STPeEl<PCcyH?<?j9u`ylQ+;Xde{QxK^o*iI4zCwSsu+<mj5`y?oJXqVlGFcQEDta%5wlS`O$(j%8(|f^&E1TsDCr@!76sZ1e60Nn#`N)K}MD2}j*=ir~*A*F&5-UaCCb*<(6?W&ItMjTVo&VQp3t}NB!wy92uoy&fd!+=5URxl(-JQ&LVmH!5JqLKL|I8!sq^unl?+o*dl4FZ^MBt)U_uZOI>=$J&!%OgwaGOzKBls}L`jy}awS!t)D!Ir$uhs2|D}P3dIAXymib_?rPf~VYd5_djW*{x*@L{fEcp-o7bE^ELIA9XaXSRAP3{eGA0M(Ei%T}uM&kWtpHl0tLcIE#sg@E3UAKR6s6>&^11up#x?LI22tNUE@k;9SZpLq4``bc2yn+s#PSU?t^?q;S~upz%(W96Hao-l5;!#75^oqAI#&dO?}eWX$T(g~hlomVtmc$~VI$zBHmQW~1FYX=PBd`oQ`38do-ShGMu@q+D8MJsL4#rgo=GSr%;35y@7kmnXis6D+>aK!1K3s%S>+~h3{4#542dv$w({sT@{B<?#ccQ@WgcS_5wKJEVVd?-DjEIGmHoHc#cdxw1^_{MpUTs~rV=*jNm!H=-+wONZgF|6805{`mFA&)({W{TA7rB~cacYbnKmIS1LbwF!i&;-q1?voYrNO8(2LhY=f1~C$(%yH)=(VaLe^`s9lOA;tg#--Gd24$^i&aCRG4y@Ej%?VDENicX@Bg4a>GbL<Lz`?D+>z~56w|bW`1A_)nCH8Jx<}cf!x52Deh$Bw~^#t9Fuz+l~p^6`rT1@B`lbU@;coZ9?yUJ=Hb|T%#NM6gt-PWAe)o=<D{?_%icyb#H`}y~Zt;O11{7l`?{M0v87*L=fL$hCI9a^S43HvVk^VTSP+e|U+II)Y%pQ3b2C2*d7)YCrBrz8|a&^}F5vT0^{7w?^+TZMc$Wn?NA!afTlr07Qdo=)YMT~PZi{{`j{z14D}pGE=<S<@e*d_j?H1KB>*YyLun+{m|_E7!-KSLtzr4wjCF!1=(ZJIjd|J429(+uMm`9<?_O5x2C`_FJulpJ*!((<nCGUMn&OQD1)NKo^)+j$X(Y#1q;&u~!uNL@Jc3p;t^3(?;zs5FLdqlqO~;4f@zVnVM}BuPv{EQ%r=asSvVf;oG;0(XLg?iJMkA?z&$k#gON>!gTw5knu>k#KVBDxvU8SHH4^j{=_I~v3>0jx4hM*FWMORDUH4b86E_+8NC8~#Y;zyybfTEkNK<ll>tYuOMb|OMv1OEUXB6O;#oa&6}*2$&W9Y=?T5ooF|ZjAsAO~ZC}{ksV$5>*l3%Su!~%DNRt<F6E}$yT8Ir^>9ux#~gI+!w?hoxqdS{qHY&2gYf9uxp1g8}-UfXC2ng|lb<(%kNltTI@UsVdSxQIE9#aoOl6-bXyci1N7{0&NPMn<TV`4J{#g%CiLEEgpKe(8Z|<YnEcp#eu%+SicemouLdrVy=548>h8p8@Q@qyA9V_tZ`~$fxK5(s*~qn+@-Z@W8rb*t*@(rJ@o(MaV;DHf;I;6&(mNim4Arb0bnT0yV&((tNLjqT$G|wV4Z;S$2gM5cD*JG(R@b5t(lU2XIac4Kz18a7)v#<|5ZivfXAyn4p?Psq7w2$3GQrcI8cbj-B$KEE86+fE`_ex*hy~S<<{~Ju>2h{>v&U$sGP*u;?*mbIGi9rKV7cR=%Qnr!l6{Pcld{rHhUuV+(4Rw%?}ionW-IoZtkH+d6cE^DJ5Tkt}(V>3#z07@~pLsZ_ixwUyphIHIyHMarjTZ#a`G^uPx-_p(SnJ?r#Wv^2HFvXPk25lvU>{Gy)@uZC(lq%h!;mU6g^6Ye~-njmP%`01~tuS9_k6nD@Mzzp5b9P4K`I^~^jDCy$D|2MQ?!!bF<UdjDnx@(gZ-w80k3o+wlRN?Vk&Cj|9ibwaohP<u`Rl+h5KNY+`<}CDXqZmr;MZnpnb@8PwdRrYUhJRWQKDS+*LB2Nj$DdZgK$VjVNKB1HfaSQU5Eam+T*Az|U39=rW9d`wcL8atycT?U9Gx$!>_!XdVs=prVKHN6&|iv<laCa|tZBl*N9@Z#iq2w7QXmSVAH<?@Sr+c@JKWv*`eC-!)j(yw7jc6FnInbWSVmK+KvBoiI61n;@JJV+T6#VZ5=GJZ`>UBUx*iF`q;s|m)4s4rm~cTz`@JGJQonty6FoYVZ-N(&k^6PmFIWyeV{H1fklqef_|bMQAvq@H_Ot2o6%%Vv`DHGuFaEQpn0@05yD72gU>)z1Yj}hdAJO2Av<gj~ACYq?s|Au=+oBxwg%0xi^y&qM>Z|;7Ldpa0uPF?{RsX!Y)fVP(p?wwa*%3?pFTA{!=<t$3)*CF*QHJ>tnZAybTmK8n_)ks*90h#wXFr!Ga$AWScS~O>D??sH9laJsoS0d~fD8VgFv(&tkSi};OWkO8iYZX-;9b{pC#K@=>9-P$HnU2!f+8LmtY^&`$)VPqoMrfxhIhm1!9vAOca!~pCHe$r(gkx7gOjg{OEGN-8K+9<qH5^LYIfy`tND{vHU7ri3>CYr>Y@+4bUEzy4mIz}z+W1bK*Je+!fA(RqBxN6n^l|Di-DW{Oto9~W?pFmcZBX*JK^%X`>EG102!*i+l7AVTqz9M>=XLE{Mj8l>G01epxsp$Q7GOxM0x;HIv|IF{i{hxdcAS}S~#>&x(%lwyhy<&YTu<NjPq9pn!F)3v#L^C8&j(<(OR`SPozRegaJd*wb_OrvfM5otiw5H!2@x*N~-b2u4!ZouYOmlT@CN4wmt@aFW3odTgCkh_u%IAr$muPr75YOh-^@b5i^gcxT@^Q{i1kjIm2pFwhPlSejVZc-7DswUr!_59s%JPTEXie*+hCg{@b~ql#+%uU4I<^yGFPxL{JnQ5eb^7jOl=mVJm!OA(T6RHHfUO_47v<PuivXxkD-=hx?(0g787`%6Vi4Ab>2D<I4J+pE}N{?TEq6Yu`i5Nf9$@Y;dTP7oE@BE~fjcp8^PjR&)casjiP8f#}QNs7LcOol~8%n)2K71ZDFpxb25UP=}Di3tDfqa5dHYRe0qQBBBWl?TjlV71;&Z-LFt>AT9n%MOXn~i?)(4BwLl}fZBt({B<cekT8%O{9!bl1cE{iZf}Sq-3NjeM&C#F=_YA9VA6m_d)CQ|G9&!NT{3#<I3U2DE%FZjvaG*4oxZcfQHw|m+k~rGGr*lco|Jv{F~sdADldcajF{`w%x~oBgREABnWShATbZ3^C0G}ePT0MeZVkI0wjY`9P(B6iAJ*lwb@9^=3o#1_;_!_Cc{P<A5qW9&6_t&VuS7@FfHd12!^E$xsmlXc_ODuySn-(Tk2)Ap!op{sM@?AKy=g4;`&!+mHk~fln1cJeoUWHaeA3*dQ7IcDQVXEgl5*DJY{HRahHEuYNR|Pq!g*t?1u2|5c-hezo}kW6P^OuJL{J|!;GTadu>a9D%aBKH#VEbs;<LK^ZFhGh#U%?1t<yvCFA7Bd=fJ&hrMqUT0hmP|;-rCR<;R0jlC;WyCGjUuvwIU`$a9_yw+aTMyD8G<BYi2p$@rWTFw7dmO^c(>G55GDH{i4WDy->{%rOWLq|>eh{gP21Npgu;ZQ#o!$y^^AbKKRmCHXZSSUW*89JsDpzu7P7fIlRU$<jMa%I3&Jh^9xM&qD7>iL9obr#4Y(=g!Ptt%FDoGU8_qN5QG6{Hti-;izuKt11m?%bR3=ckp=nD-21@NHF_L#PtVXc?6j|P+G{Ae*g8*uQ!jFGT!sY;p-(4d3c6jJbipuK2|dClNs9RP(p&S-VBcvsu)4=78k#vP*FRt8Mmp04rI3Nz?5IT*rP4!)PI@+i2C4Ck4_%J;abp8D#k<gEbyD#&K4EF!3D%;)4RVusHEj&Hd<R5D}jOy-RaRNEf(nw(Xu$x<&mvqPQ+gT;Z^AssUC;cF}sNtvLwOAfY_1U66^Rv(#Vf4Y3=+B88eN3^;21gQ-)Z)L>yF_k*g*oc8_^bGMOqzKx?(2$Fhk5fpcG;fsqTz(L3dO8mL`b!Bx2Opj9D6)IB&Ux*1t!yD1ZS{~e%+^UwA9P=HKpu5xly?8IKqsLLm-$Uv`IpxoqY{1@r}Fkw+NGUhChcf*)PWKbU+OI%kOrMw6s@8~D8I0W01`q>RPj6{kS%sEJ5amiUZk+$Y#eiL?M+XEP?Ie?hekpS%vZY(l|(GtY`Rq2EDa1F>QyeI_6+?NceP(nZZ+v}M^;axzhDe`x65O|?o3!%*FFu#^{PD7NG#)OLKUQJ6WkHmQm%9yRR4p*S-v)$QM{Wsr}({IZ!rUX$vM2h{>c0o6tDhc(!gu#5<7y;X;mSycO+6g0b8pvys0Nw-{9y6P0unP}fiuZ7L*jPdYCSs1VT?`uLsyLr%1iglaqgi`=Dh4unk?3!jnz8fgb1R+wPiMxC0Z4%&3qG59wKg@$r8B+nprFi!&Wp3Z3;q!rT0Xek<tUr}f1lI=4YE?x8ZCu!Vm<l1KmgN#pNGht42CGohy+McbT+&1qen-!NDioIu(pao=~zovODaWe_atFFk7Kyr8=Z|SEE3o8>EC+$N5Eit&t&tyb#Vh41Wt?z3<RJ}P7CTlt`$a<Y0R%G<-$$q7Hz?{+9!EE_QsJUBGKGrw#r(fxB-}ZgsztR`|9k38)ZSU)3&@c&a$wqdpx7RmuM8h#ey70B2M}6c|YgV45Pn0NHwFTD_9gf5oa?5$|AuPua<!SaE^Sq@I?n`M66QRj<|V|*tQ&YLe3Ou9wa0F*d{ixErO-R*Re=#=VNk%-?N$A%dh_5+V5m4Iv0_0Vt8c;J&Q5T#qk#Z4!`J9qrARwpz1k^A%;8yvL;~fUb?~gr|e>2CGrxjLH<A*r$0X@&v`2e7@R17C8}4SQ%mO`Qyh?CX@&I%?%4YT;O^KZWr9F$(A+4q!9o`l+&+mwb9MA<Y#J6pf>#-i!lL}Iz>v+Gfp{adBh5Ue`l!x`72xr?&a<*Nipf>Vc>gSCTHcO0e{9ub2a*udvpo94A-$$&`}>^yY|C;8awHI6ccftHj=GY~JzM24Lf{jl_b*b2>}AfqhUj~a$+$IKyEg@I)v-D?lwe0AfUTujjJ^ak2P4DapfhL|EB#;4cOAIKO-H<A1)Hp8`r2>*<HEW4!w?+()WR4%!us1oPa29mg^ST6akxa5Hjdp{2w^}yem8B*qd8=D!giy<U*en>v@wdO3BqzFm`#rPJ|(~;*|!!MBhYk`r-J7Bu1!6^*L=-OJPn(lMchAe`)$C5!%+5MPhcu}csPG?Uq-cL_pNjAj#b#~6N3dxia98qQ$@cOMfjT8rAqbid5^`wC6SZk>5=ZgOG?9=+04TDiaY3-7Q&e$XbXPuE_r4~uqdHeT7f&hp5G{)($ra4q285_2y_ZU4Or#R$5T*0k@TLwf#p^a+5Qe~e5-eykpyXS&3dWSaCLM1nTN$$A9ZvP_QrZQDOx-0+Y{%3&#F;S)P^%M7));r)`&k-9lzoWv?Cc7Ka-MzWA^d4oTf%CJNbnKMVISg=D`k};ccBUpY*|m3J~}>d>m0WL#HP;3mf)0a+1JCD6#FLq6ES+9*vAvSn4^aU~l6=Pyz#MX&Q6gv;HIpzW!Ft#snlPjC=L|YMXBSC5%JvuYV=npCa`@K(L*Meu&1uR;HGadakAc(qFw{3u37;9L#{bcKod9>eAAPL5p%W%KYXh2h(?DDp4ma_5{K|zu964t+(|3tHLHRvFaw0P+bqE+YjWITSkAUy6K_wyNm6hv0AR;`JOl+6fnL6@V#(hsSaI?J_Y;F*#6239jK;DQ)nHga}+>=0fQZ|mW9<*{;8{YS=s>g+U8!Tto50EDj(J)3{(jb(Xqqbe&H@yr=l6X>7go|5JVE4<V-X^Qi>G<7NY*gYk=)Jvs4#w7N6f0AhToR$Qs3x(QBXRz#4KZsdtozBLz2cfE=OrQwgB~2slLJ6)><0PRzaB8O_sy_p%JMn@=hCP!_rU8na2LShag!m(W3sZZu098W;x#<7yX$RS(9?P^z}qAFCqVV*B$n!%g_=Cci&#^JP~mZv=mc3U-*wSF2kfJb(gwJ;Ec1s^-p2`dcA3;_ik61@JBxT}K;;ZjP(xDS|m2I#Hf1$uw-&j{rZnc!SZb#I`f!ibD^75ZYfD@}7>^T|>PMs)VGa1Diir_M@IE;LFE8bLAwVc*G9y;q|b?dNL53CsHZRL!i@i8E4-Dq;>oUf_R@ett1I}Oa8wfgu>w(4og&Qw(3{aR;U3$WR-RI$McxQD8FlIs@F8kJt}Q22ST?1MI)gri!Mg8j?DqzZgAE68Mz(&Zs1ZEj?i6jPBaQHinLXQK?SmfIHPa7SKEr0oL*>!!l|eBpZlB2Z@A$UnFevBMZoa!DBVK3Gp?vekK#NXBU?|sKUlyEghqb?pavPh$bS~#G~Yygl0{h?KZ;yX1EC0UciZomJ!4tU62Ehnv@19OlpD37@|pEIc_Qon5=q=Jg=nCxv-f2<+!?DQqxYa}({sj=VtVR?^N42Fe*J9UBAJwB228rq)nM!hJ|7c^^V<7?FAUvJwoFU6J$srv%w($B^2xi;ZYkCm3GaK}d4Az`r+|Ldo(fqdwU81_)BYzr>B<NrZ0zQs$E8!t-bIK7FmXI=H4`_H;mLckjz~<l?Jx}vk7!|QJp6y_u6poa1&uj5kTs<ysMWfG4GQ|a+l~!S3-_0a+P<N%B`;7j4}XD_8$yKmA)T6xEGTysD7GZzK29W&KsjV9j;^#6&Ve|9?T-dS?*MJfwi26$4FzsiTQI;Cckly*FuFwk9XeO_Nn5awhCp|1%<t|fVY~R#l<Oy_KbHm#f;kvaLB&#hHx7$ZX?k$&Y{hHS!e3zhscQ^v1Izna8stQNeV4W0eQaILtBrs4jH1;HhkGqGhZx<TF29_14{ayv>E{1WtF`IO_5~}IMUD?aMph=e6IHZ?!bXWp29xe&6}nus4=B=~{1|#=b^&jeEwYjzcBUD?G%e5X1qAdD@AgPeHhUTtH`}9;BnDm&qVjF<bafkBDrQdeJ~jaVMUe~&HWLg1*z?(MZDIh8bmDKJuPZXM=?_Nh&cAEP!=o9E5}i054LHay{T=2y=%mpv6H53XD<nj1sUsLou~r?7X7kg_iAx<I{lyj=ZNzkJdH{2H)5V_x7{vWT?&+VyTZON;TNv$|=TQYCcFx}sCa`?Rf77%aQ0On}6OV#~QTpDE^n@tEM?`dD!6`#x9O5@^d2S}31VgCA+fvwb8AXd>M1Kz}7{7+i7qbD-Sft;emwBKhW=NFzi3Y%7w@zX=<w4G-7S?hpTMx&Ae=uW0FUx9im4TdXk=%T;jQQ4Reg}x%hEP`L4Tp+cGA=3+wLBKYKyPu;>*ujqPj2B&Stt(0O+sHxcapi?#ptnUbi5)uR|x_4t)OZ-Or{C-S9$glBsnqHXh4_gck*<gB_pDIPN|CHyvy+J<h?_Fq|t6dW6PLrrkDDi6FA(0mB=q>u&rW*<Bx^)*+4LnA--PXtCQ)*O3F4$3^}*xt68FE?m7pypXb~*+TiI&|1&Bu;+jo|`*GC{{XN^x44+IPg4w2XldV^hX@X_UuP}OiQM3=S=x-0dx<jIdPx#xDP%2`_$eX*9nk9qSqeZhl`p{`3fXvAs#}2>o41WWB1*ZOh=&diDveIiO?1FCw@n00zS%C}FC(o~TJl1#sIl;OaIuwVArhHT|pjC~Z3t;&5GOL{DOg>LMkSS8V%HuFDv~qY12WxU8T0VvSc9Bi_J|ECDP4zfUG-+u!QtP`61w|U~GK|VjNZWL~BBF8F$8?_7gFQ5b6b?t@TX)zz8iY>H-3m-8R=-yO&wz+0<cGKNV7mE6cK>tCq{b0_koF3x-qv$UCbakU*~To?VFN0Om7{hYMAE8P6e<ui5&awcJLjgA2$*Fbb}HGHv(-j2!4`sVhmjeb7M}>)suW_tD!6x8*Syps3X1?u#~DH>^hZ?8I)M=Mw9DmAYn8`bt=s~^vmBX%Kq&(u;iN}{dMXI)am|VFd&|Sh8*MX=q4u3zy*P{1$?wSQrzHf-mMBi@LvlXh_VFJl?Y0Hc2CT@i`^1@Zeq1KBqiH?%?q}K%U%W<%SV%Asap^u=E}+U($|^zpn&F>_6cv22L78xH-(gn59N;QD^8$xy^%8sA#x@kv#xi%w$3#v@1Ug<8-}GF^h8w<`_2SN76ttWd01rnzcA1?Ol+}bV_FD>S8!OfR^54am#?Yqz=M{L`dA`Os>gUla-0TUhsASjQMw3o|nZ`Z8tUZGfDggO;;pJ5}^c#xFxB2a|U&(H>A<3MYra=|~U*#?(Az?FN$H!kIAGAxEhIm>{`|;#)g~Yr>Dd3_I520vcLQ?+JuPv5|5L0fq*VedSso`0td4R{Ytzg{h`%a;;XUivP6k*!fiHm$E$p-e~)ruPmW<iOcKUyWvHy0HZHQ{{V(#2GQW5#@qZ|KzKOx2<)fc-v+-sj;ml{`AlYSCWXK2c7MoRP?nu<_oXVa);v8X{tgf}mVPr{KxgcFsI9Z)V2(Q<MnL^C>Ve_^pcguG^mjeU`tcT1@`U1?oyM!|3`K>{Kn^P>lQxW4>6BUNWc}cQpo`n*o}ixeOyft5z>BS)0a=ndWoeW(vg^8JL?#l3Zfq`T95HX8(8W^1R^5o-VV}{w_4fEYlp!J_}8f3o%IgQ&!V??EBLKZJGC=KgMAzbIur8q5Avshxk_>Sp(e_WXU`3YPTuHiS{b&iqlwC)wRG$scY;>X!WpV7#DjG7ZQVMfQAW+5Ai!(OXpEwvEBu}|Iod&sWh;Ose!&mvzc~x`a=;4vlfA|VnAAi)<#aU80Ct;)4&UWF8NvslKvc`$JJ3N8oExTx5Na6k4ho|-`s?%65_19#;h}$(e3P2%n#_cU|^1%K#^CJ1c)Em?J)k@DRWtt2)+rnFMPBIzkvj4*(;2p=YHi|tPQ&xL^;|M1Sd7x4yg|Oo7xsfvJe=QSpq1P!aKpvQ${7>L?od*sFWX#nLWNY!Zi$`Z2FVKq|dGR`eVrKpUa+Xju*Ku{8HjZsK?WIBV;4z<1F^x75}ejz$O)+td%>zMbcb?@8#GzBWznrNfMcVdof77vn#D<7w&^Vx2m_^+moMAUyj6I{*E8)p~cPZeh|*dCxb-n+bEQ`5W6*Dv|edxiL*AbHHC#rvZgKaXE8Imd>)v*s&jRZR0-Z#`eu4;q#zJ2r&XSZI^m~kAM{YK40RK0@TC8`pO~n1&WL!aT-pCUFC<0ZuRyqpSajxD^8!#o-sDC?feYf@G9(|Ib^>06eQK|y*V!S?%>90zh7>&3ZNEU8bf({#4B8O~g5A3Z_-dTvlWur7IT>k2bAHJipTm+as7{|<!u)(6<*mrhDH-0y*3ZvgPM!%=@tdUM5D+F099P~bjyO5X(TPxs-(QDT%L}`m7<E5(l^avo<NnMI^KLw8T9_z#xy4+^YoP3%otoDoKmqm9-wyuNu+BSVNnh3Hz2Ij(eSnY=j{il!UOh;Z5iwb&wN5%-0i4WUt8iU_GKaoO2A@D;x66@}?ry>nzN|mN2nha{&NuN_PnvC&7SWh3#<^&0{h{rI;pN<oe5YOhPBC9*F7`aqKk?hvo59WzHRuFTh5J)eLgAJ$Ddf^wL(Chmzgpwgv%q$%eIeKnTwL-}qKmB}p8mZ4>MzSFy-I24wI|M;lw|Oh?f=_`5wbldY4*mb?fhvo;w@zo2jed^VsI)_mgSEv%%SSY3cjhI=$TjtlTvuV-<VS!(KZST<_%_~B(`E<wAXtb$Eag)rewcZzI`r{c)5MkCp@~hy6ww7FgS_B!xU-FL;tz1d>pNlvaprO+jjReBQ7L{(}iF0a4ugkHq(iDAn@2qnCu!W)elr{L@^uKF&KbR{R&9wKd_gD>w_!V!{eL%9)bitxI+xG0pqx8;jaQy0kODvJ1D3lM3{TzJ|f61=VsV!o57v_e~_^$`Qh1pR4*T@dz-B^j5&)?7lBuH1Bwl&pgStO4|+LrzC**xo<A1MOu2@c+R57mfgI4xDIYxp2aKtb{;Ang3ULf4!)KSjw;q44!A<D%Flvs~fzGww+L!)30R>dc*Z2c#KG+=r8=Jp*?j~D;epqtjg$v4<X7HD3+SE@u7Kwi|NvaZ-y>`2|62JeD;|td)Xeq15Re!Ra!vcx>>NlAO2I(9vWGDp=X>@_=?a#Jo5&F~EeZIoRm*ohE$4r*}b&nX&K%q{+-z;-&mTAkh>xwCZMLi{KDv;;(vW&mNbE9aXEN<xFV}Y&xS<oT?^(7t|73z=43US|ZVBtUO^nOhNH+WI#0{=AKnL1Glm3e<4Gw}_Va*=5?Z_%oeYp!Vz_~+|GINj<NQRvK?7E%!qTkRteRU;X%PE2ZOTSTq&_pAN19d^SeAH=Qk9XT@gU|F$rw6$J|lR>BBSO7-bI6oT1N~Uwyskwu5JE};8Rk;!}Rz^{P0j5Gjk2Z=+pD<%uOHZz9B86K_c{@fJ2yLu$^@3s4(Bn^s?F6v~Bbaa)Y6lp$f+oZ@R?7QG&h^?8cCSZD$T92)jJAh%kqSGUWo5e<<=61L{<qXm0T2|7iDbYVvzTnaBR%=(41+&yNzB?v-*gS<3fzH*c;MmsQ%vD$a@ncx0@EY_x2zBKniT=c>{=oOc0j!S%NzRv39b^P`2XUcVv9DN7`wKB!Z+9nmfm1e*|X^J1Cv-ds#+EWlYya3Q;_JVc9SPhyw<RF2geqlCutKNyb^Lc7C7jU@clIk++MS3M~;4}6!33#nRwf8!?V#`Cz{$>{K?pDNHtvrY&|dcwi*}l`On^B^oKap5BSNau61XO`qD7!rkawJr-0cz`#GfsnWzHWC;uS3U_Nsh^<M<b;6dVW0Dvd;C)Gkx8Z`(AnxIXKZp8^E82jP-EKCV3G2J3097UV!t+cZPd4r3=$YS!G3ybwM3%hzWjyv-R*&^;Ri!ZQM^pmJMH;<A09G4wF+hxy-JTLU~ap<?f46rvZS@uJiXGBqJkILcrD1!816<QBVR9XF(2qkN+0%SWpy~y5LYN&`gm6@Es*V-DqnjCfQ*IKg^{^uR!u{3;(XAGswT{8khY`!7wMXj^T#qrBaj6;n9_;9(%m?1keAYDhPCSW)SM$>Ge-(ov@;0vCL0tXaZs$(*^m7L0)(-9sB((jpEez1BB`4WyQofr8R76guW>4JuaqbLJ$^D}%{aUe%#>-bW1vps}v$6#1rfpwBHT=$6c$4sOKwo{n!yh7)T*21UTv&a-!8#Bm8Gc$M9WtApvVKIgI#+gT~%RC{oyD<h}6^<%y&;EFc*Qx?xC`38%qK{Z3)0>rNT{TDO(VCecfyXdiJaUoW-tBl*7<mqCiW6@=7>mNi9eYQp{B@DWdGSv{yN;W4T#ve>pfbREgLF4svUIpM`eWa=0>oLre9onh<7}pS!_VhTH7~s_JRZ^Ohx@gJW-h%f2PIaZ)RksWRz0S|tm`ya(IC?MFa$O$4J`iAFH(x>AO)vDW~Ki6;rQ$Rc0Qo`;?auz`=EHc^OQ__KSfAhC?uXhcl4$45g(8m8T|=6rc2a46mDI_d1zel-5wf|&QEVRc~MF0%^I8){1UDX=GO~|m}da8Pz~m4b@c(cX~-Yp0K3Z4g#BQdkE9V<Dg8;J+XqcE_M=pOKM_d-T*pv3SSHN(<-D~(P+0c?Up(lb%8xj-`V83h=&Q#ZokLm|LSp}K_{gW)0GO2wC8}(kGBXpJvS-Ilt$gRZ5BLhp9pRf;ptwdf<p#@$NZZ1RN!{@O@v;{wesX|0wYV=q&*SqZvw=A0l{C#!lId;Z=q0ptkE27lOC$Sq*3bWa2oVfMtPSYq1P9Cfn#mdX!J};5^+z{fEac#}vxPfKMFbi|i)``s6npZ~9lNiWtnn`Fko<*Cbf%;yIL#G4vz@9=R`jo9VKpRZnaABlmnYzt{aiLQwHf<#2pB;KF2mnX5a!hO&oOZ$wHeW3v4$qFGwQSFN$P7)YoW2;wm+8YFq9inSqSFm$7yS+IQvitHv&+noMFMd4$v>{+sMIxmX`Z&ml+fvT%}tW_bEyf=Y`ecoM{xjGvgq*D>ZX_s4$ZC`Pg)BN@~q7QXU&jg7~X^cKUrCP05986<gj0Y?UMowKi%U_&a_1UpR&9OXiM>^ltT0<D%q};g7(e$XdUa;5C=_AYBgxt^MP1h!nsm8th$Dbi_A^h+@D|A|8mSia$kLUmqq$pk(~-971yGn~XTr8liz3{(ngqr%=^d`YL3|?WYFhE?8J}$czDl$E?QCzR!jIOGQ=ZC;s?7y~OKq4fY8WV4kFE5Jc7wo|^z#{0uhAKcbwsF65j-Po_El=;aCOKb78EY_m~MqU@AC=|||%nj7?X(WxJYQG9$Lv9$5!?1W>DB3IiybcM&S=UKK!H$A^Y>^RK5M29mURARnkMWbXZSvlll`@)}Q;8mCOaxI{x5S<e6T4iyy4zv9_jz=ZnU=f%L?Q%iIgt+nE!DvJ;lgQ1$b*CB|$;UF#3&50tP*_X|<P&R+q`_2JRZbrnlxIO4W#5UwlDIMs%T<mbQ*}$m>pt3T5c=-#Z3ssEnr3yw2*sqA%aBC+EH;{`J=@{E^+Xs5n`6U;ct+3YN$cP_ry;JGhLD1B%KB8^#GD9hNc-aAj^O3Kh;N3Yrk>JKB3eN*5-|NZ^9sU}m9ZhcLsSq;2Ud;gXqTaLirWBqPkvxBhy&^a)^T3duPbxSROBP)<f(kCW1$F>$X>}7E(J8_BS4xa&LQrfh!A;j6T@N_xDhd(kvwj`aM<F7?n5OEIh<r|F>FZ$uCgDga2e5?-wQLnaVX~|-`5vKj8PL@?)Rq(TNNQvt*meZMJ^kR)lwC$sK138ash;YH)vW5YTws#)1QJ)v~Q?u*J7|nUBD6ogu+KP)Oj6sPDkxG)Ow?f(CX1ghusgDzKJyH5Grc(%OyVp$74)z05Q%)%q&K%Ma$vughs8SFQlZN|NC?6u?{boc-1>(O*xgp=_RK8Hq9w4qqVrMFsC>y(Mm>4IkakR160~IQMs>?XyTa6Cs;Q%-T>_rMYNjw!Jd6(M&XIyq|vUKdmURJnE}~bYaWFAAe3hS5o}K~Mzuw+2WN>)WcG4yAKt|s;x4os>BP-qT4jARe`;1s@t=Kl){&LHZTSzrcH?^Z#3+Kk^Lb0B!{Vi5{w$6Gx<x&In-D+<QIm&yQOENyBO||yjvPUV+9Jq0G5PGtsxsFq?;3rXDRyWX9Hsrx-crRMakz!FP3H%8KwV=3H<=2#PKChayNlD&W=_aYlZM{BKzEgmhmQ~?c8wXX1_G(=Fr>O^=vr}TGpby33_V!u<>EvwKFJuvB*K_*Ec@B(B_^1AmMMp+<2NG;U4WafLKBP#kJtOAf}XMl#7K$3Kn*0vlb7xBfqtsA$X9oIjLhG;b!^@gWbR`4iQyI*rqFurrEyDhF@HpKS4ho_$ugGUXn=*Y2<yeD!iLl^Y^?O}^Mk$b)_0hn1`MhkI43zx`6K4S{10Y&5u2hC?h$tLdA<$sez|MkZmQ4okyWp{wkxg>`m1k>pldxX$*hMHRvq<D7d<3%XY}wFMz<Kbkj_4p>`VNUiMLy)zdY49B5Nr-k3ST!&3#qctOZ_vICA!NXhM$U%Q>YkW%8;k5QH5IYL<U~S*Qg>L>ri87zOo9>FS84iq)%!BqC>!{flqtM;IrEku`3)`IwAlaa+t|PE?;RPLK_iQTv@M_am=sL6L;9jF*)=k6CR1c%-5sk7yoTaxj4W?m$h-2}RH%fyydlYJGAul?pQO!9qKPI!vH;^dO{9z2plM`?Qg!#xT$oPS0KNg5|WJ1Q@VA2W{ALdW9`#0kb6ffqp(sP4pOJB8mSk6@M)OWCd18B=t@LLv!1S^#zVT-ZNdOk!M$eUKIOej*DNi-j=*gM2X}PC9#9m2pQN7s-$N{oah8Js-9d&7iqJax>(#KgC?hTn>*m_A9ZaeZJg`@s%AE`<8<RjRaj&dBv7c+{*@G_0`e#pXp+4qIDlNkpCDAlo@K_^gsv@ni@7v<_!KSAGmvhT>a~S4g|Oh41~AdL?tGxU5c-Xslbl8>%+3wzad-;1DmDaunKL)9bH<foGxXeatk`ew+9$rxOfsJ!)vN<!b$Ax69tsTis+?BOr#8&FmT{gU^7*eKLPh<sH=Uoy<5zcBehq_VkCx_dFmvQ!-ACo{{eq77DE#dC@6^>AuT}MiJ(O|7lsXtD*k_6*erwrpyS8KUjsE3U=;7jR*a=OXVMl2=k%H_c;YRXi_9C_peih=#AMC_i?ncr7m#qbjM*r>8&awVLdU%$1<%4^F>V0m?ooJl)w8m#(Os;fS^?E!WuB??=d3(!v=CvFy$fdi`QOFW&t{WqmsV8XAEb<DxNwVPI`1oju+QgH#<@V~XK{-48=UF8~N_v-PZ$8=N{F-vWGMohw1RM)PYNY0d()UnZ<JMl@DAqyoe~Iax*xzmS8F5!6%|#Zh?ts4!Ojg9gn{xIJHv1zo?XT7jkpI=i7a3(=3N;YewSK$p$~$aH9F!?+{f{we!x~IB$IlU)kU{?-kPY2lY1<QAHtezK8YG`Y{vF0f3#d_Yx4L+L@4o3p6vJ}MHJlv#alWTI7EvNJhk<f=#0-NuT*TA3vM=&ux)4f>_pld7PQNU%^F9z*V{U;nG82(@TlI<LXrQcORyBFKckB(Du7m1ZBHGnZ6E-}S<crTmAU$x8KChWS0`S)f%K|ICGunM~!${0MCV~}bbSnBs%45PejWa7k{Rdrg^5y<_mi{tp2G6)b(|*TcP(mtK|H1p<z;qHyFV?bVSleDJduc=@a{Uv}WO&HT=6ZBD6h#-?9;?4TRpy_f^9C@srhSkJzC84{&zLNAQx|cDjB{7N>;%<OQ;=b5dZH*ZsUco??PlK~pS>_=rGKhtqvg+HRx6|-EzSg~@1{I>fA$2v%NfVYBXxLTWSE3|T2}oXc=rDp?hf*spI5{mD=zb{=7f%m(7>c_W^23t{JF)uaE;!Yc&>kZp7sepIDGLy9{f{aV)t{s_tbvwYRQUmj1*>Ax}C$5D!%?qPx<Bfe0ZK<25{Co=GuwWJW3KLzH=B0?CP=p_mht)2HeHH|4fFAk3Sv5^h-G_4aPOt3VA45ZEovYt;%RIplvEwex6tVG@_;*1N;BZDZm#$n(zK^sHq1<oz#Ny^eCUn1m!<4W7q?!qJ-q-K4X{%R;xS|%UQRWFpsL@t#B!EIvASxykjg$4$k*sLlGXmf^ZC_rG(k&<`ArzKjg-S%A38v@yDX$Y>vN$<>CiE=`qAiq~Xxa({JR}y3Qx^@?{%}?;hVRm>JsA%qc?G1Wt%UO_ar$Gn?zAtD5w6#xve}BvS38dR@ZEMc;4ehW(Rhb3&um-L+!#;T2T-M4RhX=mTL*L%~8iDa_;cun1WOAp+K!nU9D;U7z|dbk+D`I-y@TCD_zaR<rCq5U4cM-rWKo!xudn<21!byvmFW966;@O9JpK8z3sglP7JolttYrr142RN_8H#wTDY_TOYFH8wnVaqH7LE|JMvz;88$pdrs}3=|tkAZ8|Zxb51CpwO2|xa$w2&wy0a>;;DOGyC+B|;5+r0pZs|DrWPx2)~E8<+>P_jr@W?6tsc1n_I`UKE5-8VO0F%P%J?9TKMJ8CQh9BAIdB?+k80O%9j55+@TWIr+P`S#xcF0A7{5{n{kyj*FjGCr-P>e8UID}6ITM;};l{AfIoKDd4Sv{3*D-5T@cT<GKyt(+<Z=-7A>*y0KIEgbQ+6-Eg813Rg&q8amwznwN&fyLkZAhlts%hNfD5AR6*<GT52d4^oAa+4=YNQCgsb$oMo1Db^<Dj3uJ4XT+7oRJBu%D2`5v}Z0(D3nVH4UhO0LI42Xue96}Qv9m8;?k8_E55#%OgcNW|<JFmjkXF6odJ!5G2`GiIv1Vqd(k#$osZM4QCh-wP*0tPW75%u5(80udvap>M%PG$;nkAlg0|p5x13rzR>V{YnLRK<LvRS(}G*=j3Wv-sh3}@=IqKs?{WbD1}W{qTZr|SfViW7r}QKzh5~fDXFOL%;d1lbNsU~TkfARjPwv*Nve^R_VkxHc-vr?b)Y2am$dl%;;WNdW1YVW@{MHEm7sg)c`~s3d^7`?&y|`Pfx9F>utUi!tQ&K)UiUr*f82n4T5>8Bpxy_1FbSXp^I?x6ZiG#txB#(^!ZU|#I#+K3<n!e-MP3^XP#g5ITDb}E6NR)l7?b^G9$4#OGYTo#hQONkg2qDShvFKq1DQb~A2@A*>@*mXF&mk$n&ks@Vi?W45k;$+&V<gV0}a1ii@Oi(G||@nuga>p=sf4#mpKidiW2n?tJi>V?AcOGyaS!TG}zqM^Z!0zO5hp+MeM+LR^4aj=OYkIv0N|9rM@R!F&d=JWG%m^)S}e<k|S==q#ZQ=ML4c_YAxgAN6KA)rNT{S`2~oXK!!O6XGD474c1H&!dC7&aK44Hrt|a2^r!AB_Z9?c5kmD)Tl7VsEq|~68Okm;w;JidQ1kZ_UgRNG*d@0+;$~d?Kd#k`ocLoJNJhJwDGW$o%_ql~jB4I2L)ss8I?VF5ICvZ`ycvZ1llk&C3OO;&@2tJRP%HilV_TvtWOcBGJ<x4QV>g6`_B_<YlDL=*9*%wq2GP}^UhS@`c^}i!Q!ZzMK5JaKW+vI;^!gfF#D~w3s=<u!v_4LA`Ncgm>@=CVEUivP)Kv@Jf5`teF*UtA5d4>OH7^F-IW?Kna<w!@2PGsCH9;C?QL$NWsG^bqAud!*?rFqziIzc-<A#$_1kK<0|E_N2gKq*QqN4r{)TEE6kYLlcN{c%1N|B|l`=!WxWX2rPc@h9-tDLY-VA4ygo;s&aEgG{%Gvw<}MF*uIo_A8MKSw$bCwPXhRSxA{&0B%GW#!c7`&3^hX*c`}+$3)}E;f+PKfy7QnGJ^4$%aqwn>IOhXyjaG$l$BPqJhgY>&;*~o!t?4p;smI%`6x~<U|D|;NOssC0-^spFCo;prjU*;X**?0MjP{`*n{!80=Y6;9cxrSR2p!e<hK|rAirV>KE|!2RELb?i#Jk!{LmmpMUs&7?*ZN7IZ_+k|x&%OO87}=D%jIsbj+nr2~rmvwJU3Dh&~#g*@{F&P517&FElJx17qNAv${fGB8%D!cK9hLRk}!(Tr&FY}|YY_B{GLWQ3l+JY!uIBXQX?>(6TmkqK17916@?5J5um)l3|j0Ku5RzMf#8@tc1nc4j|Lruw3Jb}7s4&sB4o?N*ojRPMUq<HiHim`Tj|{UGcHKmQ^;HKfXkOrDV#I40y*Dr=!NL|8ryHrY<b)?n-igNM4iSR2J?B5dyeFLoMFTXV9FEH!#|LxP~$7yV&WA;WBCkbHcW@dVS+tnmDP>=L9t=arUbh1E4!LqEMrp2mJ(p0(=%kp3^%z=T-~dPn3DHy=9dv%XuhVzI3WIbsL|^&SB|>^7WUZ$uCwf?PW{xZ)RRPc_0k;)%(4Ow)`CJS!HHCxfJ^4vTe5Dz_{4KN%Of9cnpMZ_|j3kv~U&l^pEdqv7lccIRz7q-^{D964U_rSi=9$TGQ6Nn4FoH(E|`2i232xy}1q)>GC7@Q}q9e}tY5`huJKwOOalBtoaJXQRdwx`%nPY~KmMeEb<~UYcp_jm9g~POwOf(i&&}c$A-Y{J*~}fsJN8y)N1*-|51x+t5bvpo+n3%B?v~a^V}P!lWTtK52ovp3WWS1M44KbXZ~<j|u#6q?3z9fP9`-f$*8bId;q!Ll%&%pl{SV1CI^%`a{s62cAW8N>s44Ve|0aQI9<<UCF}n)V1{=ArNM!m{?@GelM?^wWdjM?#Sek2S{<T_+yQN{zj5P4cW?*PiEww)!P*;ZQ3ig9Je(wWzgJq9V8T$k<eOu1Cn&y0Isefk3)0QOoQnInhir-mp5IPY-JwL4}7!O9&+5Mm>aLY+>DH)!P%mgGyND4F0hWk%HDQzRbnuD8c;GQRE0FiQ_NZCpaEbQVcdWUI0$OO1+K?=I%An>S{?<>?bZJ)^~6mY(p#TT7oXa1>Di8XGrm87yx_LsU;RX27V|&E9R~jYItb;9`6^@`2>c=$*U>i?z=uheqkC+I{#On`_04qIzfrQ^zrfHev|n6*RVpkM7ieI;M}>@Wa=ja_JFm*yBcJO60GKqbzvJrbCS?!xe?{5CfxE0}8yP8+3MqVV1JYcjezNNP-hOEOwPqHHNE|=_Q$hVx&_Cn<KEB|1oDQNK7MJMS3S4|lImCH%HT($RFE93|4^}Fl@LA0hhq#-qXI^D8BmRvHOFyoqlXE>lP;?S*%x4ileuG!)l8xovdM!1HTfU&ErcLu+f9o}QaAAEtU@)>o795+0ljsu?*$963qYeA$mefFnKRzwtW)DIMZ+aBTE`M|GpE|jQ(jcxVQ)D!!wp~9G014m-v#TWcY-(gG#5@g%nRg`}RX^9LFG57Sh;oE+u5p&^%@<0A?63Z2$N?Atbuc?b6l;!kd0V0z=Awon`LO=inIT*Jqv$+(BUG{|yq{n3cC%nXPK&o7=S<Tyn-MwZ%&#9TS+Zp8rl`92obSj`_#>%41mz(Y!yU`Z#40;(b<91;`mF;)67f^l$WZ?{b}LkPhS8r?q&a1ya=q$=Ag>jviMFU~iH@s?)h(m!&QeC|Xw5-4Atx{G>rR02?AT*qRyNH09)86K^gS&~8p&Glp#Q?G&H147Ca%)UkYM6ogLvb#do2f_RBPXHJzP;HPA|QQXUb&@aj@LX@y_+#+DL7hsSDu(d%Ly!YRNe;V;E2>V$*3DPw<?`;O2gUk!&mmGTIO7IkJvaId5OIj)s2Zou4h9P&ye>1IE7;RmLJG9Y1Lz+PTJqT0HzF<@V5Rzn?dXQ}^r(YlX}w6ddF9nnw&DTBW30`Z*u5&8-1@&I3(FFV8s$z~R8!E~!e5s+q*?;#{IHXR_f!t*v3RBZnvFk!4w7)VbiYqSTP;HAq{BcYz}z-vm?0R{gQx`FMk1&2NikHDK=>U%y!aT*(??P9u<HT;y_MpKIRmps~7I1dv|ZMPB*54yJ-IZUam~%hIzdj%sH;?ORoT6geNFA0(yUagp@2sRp8Wsr9YSz$XV-Fs0H|%0#z?7Rm=4(8;yA@<j-D#4oRS^S?q&ve__M>FhNj(tFD;_2z!1Tl9-sAHUIXMf<6Xhy!5pYYaG1w1QHk;jEuBasDKMfx?qWscj}5w$zKeOIQ6`t$T)z{z7CjV|Igx9Z?}TV_}>!`4I&54TD){!bE12m7#f72S-4uLFQ50e|smqvR(4cFB_rvFf@oW5`+QjV(0T^Px0@;t{*lJ9C08>o+Q6?-<J!Uz-Abwm&9jA{hkbq-;l{fikq9P!}GseKj!vTe@{UhyX?d3`%$T^IdGb1Xacb|Sb?*ua4wESD76v6xJ<Ia8BRpaIQp?%rSGepnUj<u-1;V|8Go}jMMh#kY9Cr2xbLULu{WPE@=Q5Ol<b99vc7OH-14!a5Pl3RXC!3;If*!tPL9xXO!u(ehaFcnA$z2-sFa2j>Ur*0ed=f9<TbhNc^iY<uI;UNouV|(*A<tCf$uK5(a1zVq5HMZEOf`pIEDuRK|BCzOCHea`%r|j8|y^@;2|6&#oF-4D=q>tD+n0WQpZ3h-fi-<L4VPd0WdNOfXN30$Yj_dJUzT1kwOxuiUT-OHxxduD;H<zgJ1%5fqv-kp~#J>de=_IWoCK&*`*H(qAg6L(hNV=1y`9zZgd#tW46Gv9+~b;i=p1x!2uaYIc(Cz^8!eV1AK?ZeX86F^$P&K0-2Hq49WZ@0Es>-bO2Nt_7C^q783&a*Z}z*V8mzw3O)xYEJM$y5>p=rq=Y&OSSRdb!C|n&_}oMaIm3#ykAV>ce^c>=Z7Jb)^&<9TAa#H#Q4;X7?m#Nq1EKwFe9?J~6g+0w0zHwQOW63;ju<sWN4Zl+qvh{R@B;z0YW9KEMiW}GzbW9lX5eqsSEQuWbPhK;%>k0l`W@D82V{Xf1XfrweS_t5(aiAUS{o1T6Z{o66~Z$N0mmS49`U-hWPgF)CZN*C11Q7M0Ru`HI0TAC9z3&4{&CJYpKDzQPWmtdI=T81Z8{>t4#(CnVxd8O>h3gw?uY5TBnH}JfwB^z!PEO5GVthRLu|GmNir2-Xh;FSiW${>E$Rg@z=e$8A4wMf<5iH@Ig~bRj3IP{st>rYh%?7f61h*XBtaf7Y<sXTvs^A=1On8fE&5ILOO}Fo)-~5r4H;~ZIN2X##wj=o2lkx^^)0mxj%A0{pdg5l2a#SwC<8HCVsG~$=p)0RNsk~(7c1DR1sdcxU*9?0fmyerH|gsp2@DDSJShl*5A=(9MQ2aIE(*%GvtkW^5#E3?IB2;&fahoTz<3-~wj|ewUi%k;FVpTk@Cs?bmOw7zcj^Hh8|GI5gB_hNi;GKN&j;UTVQo#ZXh8;;G@1w{6&5cx4pA;U&YnefrZ6{o<qFQv=KLO#@})<kd6?eJKemgPF>`3lZ~v$h?G*CB{=U>G0JSDa!25mH6cV5=b5wbZ`2?Wj-w&+zB(WEV?t^1`)pcHe^eTW?D|Enb0iJks(9ae`pr6pw_S3zB#PPuTgp^y987UTp+q{}E;njiZsZRZ3JACMn0bV?oZkU1H7YjM{oWXs93kUEJEzNV$^^0Pe4;UnX1J_tdbyz_hF=MKXQ<7C2?+YTl^zzrD4UC^Wf0;LXSm+}a(?Q?Ib&yvCHe>N3=?V(mRe-u`zKxws;D;CF>jG~PYY67&Pt4~@xR=$_q8XL({!IjQ_z(VFQs3B+&;sA4^j0wi-qcxEY}+8x0E6Ix1c0h>hmGz;f!fdy1c7>iLzs*}kTbBuOpyKc6hw|Aj=>J<fQ;U=0p5n;qzCq4xTKaT#fgGIv;lzKLzCc=Wo$;X_*0a&34pD9(725|P6_1@@Elyd?0rwlv%lxT0G`N@12PbTfH4$k`6?3Aqi_=x{sk__12f=q=S^IK3T=}q7zsjNcmNx|`WR$4!%6bqkNTokJBrqBvFbF~Ge5ZJHnT8E+C0#AG~;>}DJ~2k;DU}M0DqYLD^T#*ZAi2r9-*}Nn%=_6!D6DBF;9(`nI1Spo8qV$%%#BMcwRY?zG$Ki?m}tc^b3JL!lg=oW66<?8r3Pk5r{aTLH}O4Rd)<gexu`yNb`>_hAE<{^e{j?;@>vHqgezG8H~;G$PZ}Fitxh0H!qU~xk=>-%n1o%c!&T9D4=*m?Wd&=>yv|^ti8*yJY*Uqz*QWJ4dQu9zjk?mC)WX@{ev|iz@_dtYW?wP&0{&5p}$>_eXjz<K?3x_BD3)i1Zwlt<;$Tjmk*>yf4S|w(iJiQr6;|blGv{V$i(lq37;|e>XlNO-f#0hpn;yTg@}e<sdge-Hy|E1A^hU974l4zfOx!;pe@eLlWkx>QKK54J*iYVuHroLKvbR79L?0X*?<ywU>r(0-=6lG&d^V@AV#2M>yP9Wy#O3t3=|+ri8h5|e_I3Cc#B^ovp*z3)k_OJK&})%#N4b;1_hHCf75uX6h$45Q63=HgvF_xRC(^DPys@*FGB*nYiN)q2mSps5xnTlcJu+HT7;9atN0eEMnwRKt7JzZ1$2;n93vhm<#zXqne|W=nLB_+scG*e7-gQeMewPvyfS(fSOdd!#vRm#BsN(8WI#JXQaWOAH}#NxbshoGBBTjWKNho%I2*)dIlDoEY+MjD!3eMR0!5#i1rHFdrM&qR=i9HtVYCe3(?toA$3hfG2!UW^SIak~zA%`9Z_7?JMb%vGlHI-yTd7WX(I$-!#6o&g$AAa7Y$x!!ot2jAjKZIV+~MGZ1P2JdP5eFLd>}`M4;FBgB1MEsF2TS;`z{flLGoLP-W8`RdF{zBLtCF@<_E*0l^CX>0<$y*FK;e9`^00!WVC`rYmR|_6f=!pXP?i(GrU*94ZM*RNZ}1gCm;Yn;;Z`22l6-lc&qv#p(LDAW+&f8aD7yEJjdPZ7pJBv(cTpZQ8KY1!5pwd=m5Z$<+HmfW}AHVCbn9AnDbzr<Ds6V0vM0x<jaRiBD9Ao%j98t%{3gQx9_L)V6Xs1$VI0C<xp^N2U`U6d`gVJ0CiMueGso}&x63|I9tAlKre%kH75^rJ8=qZnHT1Yvxkf$<acA4<pbVp3S^E@Rz9Lpe9{DC`H@7Q<XHPc(xM$GM@XG%1=ou00BstuUi?DQvqkC%4spDoQ4lK*qdoz`Fqu&d<<#v47_JNYI{J+PPSB=~7RyoMqzi|#z~cZ#d<797^6b3_Wd0%b3E$xr9iO*s(2b&HmSP8dTM9ZYAaS)<4_JBDe%F1VL0Y}<V)msS8AMVoZhs)|qCoZNMM40!1rD_C!AMyc%qPW0d;&S=lrWI+v;vzyyhL8*b_O>pBb=h(L3-bdL{6%4JE`JKqp(Hckp6*JaxU~Y*-C&<&7$rfY~K09fT#^59n_wJ7<lv0v6&?blKsYiNpzu*6#?B{;3(01K;`-!_1ig@34+xZ`h*}je{HfUriHme;4}5aKp$m7!2^w=B8I0;f$oqXlaz2uXFAMzVfzcMbSE~S`Lr)QaGnqab_ICDBnaw!<kIhl>hGL_DL-GfvI0AJLOuEu!bSrb>>c`W+AJl>1R%t3aN>vO0dxJJtWfd++eV@#1~(x1dmaLA!(GH^uzOQ1PW6^dY^e;Gk0V*bxU43Xff(9e*jwtEU>^k(fd~B1WL9*4clt4J#2y+9@n3kiK^>xq<5vu_4`fsyf?wRveeiqhBz}MZ8#RyszQR=$Ao~zT5g^~<>0;UkZZ$fD^+mkNwCyfLK*u|H#09LBNBX~qs8EpR3IAotm4Q|1J;$4$+R_++3@mVni>4Xu`vRcBA^IMW@>qF<!i2mL>A4^Y{?H2W<r{k;e@$@)B9-T0IsCAq{NTvztq%TnGD)}Z??Z1U7t3M>8O7p15u-<Fori#4q}A~Qh=cwcB}fM>!@ubhPuA^vFxXS=Gz2`Vv+2D|2-7=g+pgng!z6nk7Jy9ifNK|WF<O8aZW0rGH|7xrVLaw*izGf1C{NHW6``5`$Y8P#fRE`VOGw#6IBSh0ymuV{*x)Anro#}BnN|bA@FvNn;Y9>T*$!=KyGIO^mYI|m7c9&Y5}1HwxC!iK7;r=+ogf}E(~Obl0V_n2w=Yjid=S;`KyHaCVK^N3Mb(Q?fL%zMf{wOu*KrQ4C~HZM6XV_jYU&&e`D1{8e;}FKM|2!^x=TrxTK0qcSP_DeX@nnepfzND-A-^48yU&~PYWBQAlBD?6eSpp<;t7Ob9R5z=?I4S>l+o4%qLjEg;CEm3X}tk1E6R#9Vt#Iy6hYMkOhO?;L!o}x=uw2+%*4I@IFPVjD3j)d?{g$i-o0ggvak=*r=w+?F+BL&fSiF0AXl>drAV-A6#^WmCV7b1I5|WMghilHqefI1;c|;NDeOp2vRRZ#?*cl&6_qU6$FXy0G+WvR*)utegEo|#2zgDYl4%`0sA0P-3Eb~0*X|leQlnpimD#m06}s&7iAtmT5R8O(4+i9K!X2FCMR`jp`Xd5b^i6-J;4gni%}j00X_p!k^t;vRTg<uIsj)8r{e)bf7#&YzgjcUe>)`iTaUIGNNHvPGf@2dAc^c(?9@J0N(K<!;8b)!TT!Ka4Y@+MzC=0Q=nn>V0DWVL;26`_+VTjVMdqiAKayQ_v!oN5q|*>8!s#gnav;@NVjHhC9oj09yTa?>qggC%LYfyQy~bM9RDSa+HFEiD0X6qI8v{>&VVrSuKW9RQ+P}g0^}`Eg7?qc368-oi!69d*cfZV(<t9s;mM#vhtyR;jZ6A@Tw<~}vkMK52&!_#-EoOH<yH5cH_<CNS28AS90t0JcgsT@;_x3vsnSd2}1E&#DH{zmKjWEh{lk>ro5&CLB7tm7sRm?(#R-?JGIy1jE?@`y-FJuXp!f9{Z$1Mf$OKLL{+s#vb^Y7IPY#q6!SRv=02D^&#pUVbS`6W@Aop)UMuG3-9HkT6G$AFM$hNVhYgRxa~k5^(CoQult6z8hiLv(C=wSU%`5VX>MjJYA%n-+f#+A-}Im*OpMn>Q--d4gIcv?W}zv@!{8T!DfbQ-jvb+tI*havx@RIC|oIAM`mXt7&Fgr^sA3A<8B<usKjhG|U{`UMEEo!uE-EOPwZ;`E=i~=Ljb_>%*H)IwD2o5D*8F&X3{2X}h8LE9~GDBnyekHrZ$(%H;8MfRQu|2ryiiFH!)pWqmk5fvd`8m)DMT;BPM$2km|8gZ9xOD+W0os!p&BOz|96<~iPTfDpJVu6CvKxT>v=5GzRq;xf5usYm|?JuYtyi5%b7sM<XI5T2HDr7mU7QL<YHVpD;!32~d*dZ#J>^LG_!wUYzo-YPli+q}~1p}oy&9R^>vR8kFVwFdIAiQ3myzbFFu5zyRAj0i`r>^;;|NyniYQ#g?SCgOg3O(s}V+)0Sx3QPuW#@+#~Z!Y4aS&*RhL4aX`)68xz`t4x)h9C@dzAsEMyKm<kZeNwCY{wB54yiPhXp)t(r5jNz-;dC{8a8pCRdajDci+13c6u5$vNHhTMyU_+#`3gUVh0<cb{l9@kZ1F=GW|v?O_IgUJe0{!ObdKP<D)I^LOKXyEP9tlouiF>VB2>WND;EEl#Jc8b8egs8A7+W7YDNJ9kS&-|EgG4h88^tQ;?1s)}_eDtFvn3&)!mB-NU{`+q7Zbf;T&YYXTa;+DUf&)e}wZ_96cCMFNvll7)nzQE5Vj`Hi4u{&2L6PF`HuRxQ`Rrt1Q@<h9<+8Gz(TXPLhBZhoid=D4a=-^+H5h-1y#wAL4@1!qmBNrra@u7}<vq1`y9MuL2y%`Z#dkx2C@*4-^4Vt#BKu!(d#^o)Qqr`7Cap4$p#6ol*Xu2^l_`K;H+d9%;TZ!NF2!x;S>yWgEr%HaxG!hbvclU~}Vt6xnMP<KeAwo)=6*qASd^pb-cq8vrghJhu+T*%j)x(_WYP@G1qzTTk*G+YX-zhvBW`KvPavs-@nB5K2|&oOC$u{5$B@Jq&86|-q|8Z}BG`=JXUUjDtSJ@e6$;PcEM#Gj(2q+-uev+p%67P{@EBBK!EmS?}&u*cjWGpr>6MRD^K-Ca?wFHPp48V?1GFX=hu_%!VLntRZ1?mtTY#^D9!n9kG?0l#z|tkK3io$lwb(9?TFQkiOZziPVrC728HHGOcKCNfBUi`<Z@ZubO!ho%^h+?vb6jz9CV^6l^34suW3<BXkl$1*h>xsyh}*W6pa)5T(Cqg=#Ar!RKFh;kR-Q3}n2Wh$cA&BDc|v2R~A-@Wz-+Q^#B>LyaiXdY*14?Nwg9sS7Ox`JoGS2Kx%stL)rFP1JH7vUQg>*CGSwzw`o7P0`I<aDkBweb!pxZ*H7KAhN~IS=7WNd5O7tPWp4aRk|Y&(x<7uTZVy?dFEpB2lzi@a)xt*=$5QVN|7Uc)NmpFwn*&y7!%w7KW3bnX%bFJ0GP&0!Z=420Y(audy9TZ-)l1d*dj1cn)6?-K<)qG3R%8owdr(8C<JgVM8rXhn<sbQ;)NY`HMtWPj2lKRLsGK3zv~TZBQF(<{3dOO#36l)!&b94(H}(#Hh{MD>Nrg1*H5LGMv~$(3`92YvX3CHFVsJhQx{L`H~&<Q>$$wtV)IY5tLaHmsGNDDE5$DJYov6{A9x~vHkQ{B$pQ#4+sCWr>@n?X51nxcoq7l<5aCfoj!&RmfUzlW7GA*7Dm~4<CcW?a#-`t{9#TlsFk=Kg+z7JpI|l&X_GOOnGO_IuJg-F62YKfjuo62xrUkHpD)`Xu`5okXOf#}nN~2@H_>i^jtNhB%PLY0CVm?8iNnmukn_WU9A5|s5-8lsMbPIg#mgEnzZKR!!M|vcET)9KZX=eMIzDKVHPvsVVUoK()vAwi3#@I?5}|tysj%{L9bupSyXEIGyTVLvXwXhaV1AksnlVdc=VSj8o(V^LeeYWq3bl8KR(UCYRPWnMyuu*~d9-XN`-M+7&Tt5DOp~SC<OB0DD2t(^l|rL&q`g(1kB=VYq&D~VJ{<`5N*Uvpe7~ge;B0|bc|phc$}Sz@>F$BNJy34Z(tWI8R|0}eE~mNWLwK%0){?z&-|8QhFLh%CGjsHWT~uk&hZg2nkL%)Ho6Vf}{&0oHH>Bh3K-hhQrv%k$Em#vWGiS2>wrYOdCV01=(ZdlbShcRqR4>J28^GhaZmqQfd0!G)pZqZYlRY09JI=}WTFPh|`tL^A3(tTH*~pSXe+#YyD5a@KQ6hMJ-Q4<OEKc;2!mCqU#Uy2=2Ilr#wf=S&{0tWXER!O$#6W`S_JSxFfuV?G)@_bHfer8n?xr1NnElt5cfTaysMGxR+~=$M94RRJULMS1mBzcwK?qcdxLzC&C68^f()PmBrt4lYKf9OFvIkS`Vm&zUhW%v4;#Z%)MWPGwXTdG9<mfN%ET6v87k)>_<942+jD}D9`8*X=;vv}u4s2xZvqzt$eggn9ZO_qxf@Fwe{RR-DD0A;vMMP-XlKu`*+rXk=q=5L5KB#oi(F?D>$EY7DL9vMI3iQpWd)n_=RC<+`Zw!h#i8Gjv4a<iJI}qCN-t-e7QCrKB&r||m2kv85`FRF<TI*SIjB?=h`FiYTXoOO%xfiY&7EJ^$RfuZ3zO4lpK2%ar)o2jUjMNfRU?;FL{B^?<Ji8JciUCkL4?ouhd#M+Y&R!7KhDI%IBM~D$XcJE{)^%SWU~Xt{%3mTks_gieN^;(?829*z*~m`Ni{r$@qukH;#`BKocLg<IpF~91C>>R0IPeW{JZ~7`_nqutLBNe}!y~76T;i7&8%y?OVo^1%B;U*F`YJ^!Y;}NnOpvo5o^<mt8MBSZX`NOl?=pTn>5#m?hgd~$as~nk8<M9)y+U6<7bvu%e*kgKoIkp9vmDT8C9kCg0i|H3A*CEU9iYjzwA+XCSxc?|-p1(rfqp~3F6?a+bbn(TV$Y7}HVb9@ZWCf=j=Bo=+jsHS3{CD+(a_E_IHK6`<!u$5(%Ahfn^!wex=JIY`Z1wh`<9Z_gPH`lNF|g!eNzi{S-h6bGhEZ+MXN@=&>Ty^>gt|3BV^I#zu(pKd+tJ(z<#gCV9s|1ZMCQ|M=CWv&C_1FgPjx0&*f;2C%(a^S0HJHo+3?S;KCNj6%~-Pmhpp;0*2YgQYavB>5(1lEY>vGlW82yecza!;I40B(1vb(=%!-<?~s3O-~|Mr{fmLkwzhN9;zO2!vdnT_5T9lLd*Js2E7Qx$DRdg_hYK{9C}<$i5zcMj8hA4>NvJB}`<5#5t(KP53Q7<&=zjw3ulM=KO|*e|AZ_&38EXn1pf461(4CN)QlRNo8}<^DNnzB@GYZp_yF)h}og&#YTk#U7t4xibu^g-W3Uqi~{XKE7`cE5hrDX_je3&YH!4+J04onIBwJSqvRGgXK>LCh+pXd^M$E*SAevdAt=YvjfRvrhx;Ab6Syd1y8s$pyfvM?bFRh66CZ>4;9pu3xK;*FR7J`Ev7vfI$)5b=J*ms_{$n23P}+1@l4%%g>qLN{of16}d)@WN%LATq`pcnSLxUE^4;#Usc<NFE{NBq6?2bp^HQ^M>t-MJ(oe3yMJ!$CE5>^6-R=^~Xm-xDLJ#d}WLCr7j)`U4M<bl1w`hG8_G;{bn33zcTF}IVtPIB3?-kox;g{N3R6%ilIKEDXDGS#@lzk1!XC$m=01!EExp$hvJD?U`i)lHN?fEpW(CJwm}<f1t(DPAz^aA?Z^9k#Cwy~FM?@SE#+vD+Ev)8Xg(kfb@t!S7ZJFlbB@$TV+;zl!z+>B0{*r{w%5i59h(S@@7j7Rvo{Lq5%U##pv{;Wti~AriE5q)J<7mIoJ2)Z%)+Rvp6nN}=oD&G2IltZy_T=1YWRr-FDzd98S=OBs`c1}PjFL3(}Kl;qTAN=F52oCuB2S^y^mYEr&Is=q3&RjBN4%8*LAEv6k#d8)em0)Aj#{R#jjfC@>z%)u^@O-%n7!Pr0bh@P}lmb)ORAxFfRuL>RBuPV!L)0Q=q$n(TDe5;Kr-2Mob+3a&!+@`7`Id{N@y25jW=fmKxw6kaI~WN-oT{e5BZNu}sQE`8Q6-nla(;@po%PfAA?q=%}vD9Oq0fq9U*mqx>>w!6rHy(9<C{G5(`A8)L{VWDG}oJ|153s^GXlAYgy{xWh+WMLb!b4=?3VVyX-ErYl$0yFIuMaO%DaV-GiESfGmt>et>GmWr^Wu%8ts%Ny!mV*3RJs)ocE3``?%#jvFuL94Y>(9Dy3$}@&3>eGg~8kXjD>pJSDhI|-D(Ehy#D#{#{_U$#PGF5vgBgS&YTWA!pHJjGkQ+_j|a@clm!s8oY3fC65k^JyQ9c4g~fGJ95J6PRQ1{Iox-86|)M}cGff!{V))1O}+Rn>D0;!BqT{98K2?JBxL-V@H{>V^g+EA&Mdn!iI^)hK?NeMHZ&i_gvb3x~Z^e0v<~1GVGMfc)$!%x{-)_#uKU?sa{Lj?z1`(>APzeOwtEFt(Sz+%g+WvLwGV(BTPlB$^e|dhhQMqoS@Jlj2gV9b1b(vhPB_>$CrDp#WTd5Y&raT~6i_`OnuP!QO|QS>aEm_mfpY^Os%Z?*S#<k{%skE$WACNF+|m&Cr~ks?L1XBzpAPJTHY99OLqLM&=BL7yC_GJ^E2H=?>+RzckcC>4)H8^1_Jq%#}|J245`xa*LRF6nCHq7BX+3FR?uZW%z&>oo6`aXuG7~kpooV85d2FqW<15gH7p=3D!;tq$X3Xp0*MIIP{VLv)4rLLrzkF{aU>I*0jSBTT)pT251O0Mt?6f;^-zUrsVHg%+Maon0lmd6`VX%?YviVF8EtUfc2<$e5Z8uGC6#$Pmjq>&+RV|c0?W?pmomD_>&cuPFCtj7A!LfvUSk~h_%`092XDLQ5C|eyi|1nh+JUIa#gXqmE`zBC{mX{orhhysfxEbO^Pjr!YMyJDkYB^G@ccXn-)ZXLLxOM-lHbyi`;&kjIe?XEC3eon>AVg{qKLc|Jlq>w;2APfB*S!ZqBfUY!5=!E(*5GD$5%al%^7C-b=bq)A#CAPk?8N&EdCS{`_*zCG?!hFrLzRojw+60-nX%!8G-hPM{$zHRD;`m5^!s5J2y|O46GS8qfi|k_G6w^jYDfg=&Y23$Wp3jHGk+w!q;0T~q>sM*&J&_lpi}6x!db#V15tCL^f^)Vmwo&WWnMP@75T3i7-;2uZU*yNiDX&%~EGJnd6TFqiWI)BVoId0$E}aF-{1dThJoA6uF3j}7}swBHgqKc~FtrB^pwB=y>_7U~szMu>_%4$y_2Ypvhib7K9HnR361p8n8W<v#%JQ}0>oFK?Sp#3U=UR3?}P++_BJG$KnJl@X=V2O~pz&!Bp6SF5lo=dns1=zkup^u~sP+y>upXmr4<2juE0C#p2A3hAuz1e1{6aAHiMPkfc4TD*VUa%TU9Ehw4MQ%NxRN%1rq2gi(_ERz<CUayDSr&%w<sDyu`MD1tGb~I@qO-bKTlh@><Ld53;xdt-s_9xl0FW~|zLhV;r)1eJ1{)C2@V;DM-{LY0|kbTIC)tImfTW*Fzm02?XW=#=kOzHNOoWZj)qu6}p`n2Nw4=eth%ZN_7jexl1H^Il|t?n3!E2qyFdx<pRz5~`?3qL?u>*6ajjLZ+_P@Maecth?58%Pdhxm~ApD4Rkq<F(vJA?Ep4YJ|Kn_|SaFx5w7{O=?7%qK-oLM5wXzhzS?9o(dAG-E6MdcimXZGc92lP6H;t0(|_g>qToNU>NGf*s9fKnfxx2(d|Ym32nM0n=pQoOi__+67-Zqd`NUlGfkR{pWHXAoBmJyA^sO{n5WN!j`O+7s?Ci*gF}OJ`BEB;iY*oRRF&&Cg6OlxG`YO?lR&Jku%hO74`EFX`4wSeuF!Th;|VW9#1|u4YM)BQ<hU{zvt^XYQcGkm?fLV!2jt}%qbce*KBR~c9xEy1I)rJz8^H!_!kJ|m%@;?et1wp*gyzc7puCz?O-Y!<Y}U-E!U8A2oXjc^S658gDInvxj8sl?Q0U3*<k~rF_IRh>>>Krmue`>F9!lwI9(P`-i-u!bU<ai~eC6RlziFS!=yN#hq>_u#z*ycH>A?J8D|fU=JB_xaN(Q6@L(?siTmQbcDrAdDo$hc9rTIx_9wJySoeCgSvi9{LRv1}EaOm(JhrYc(h8t=Qi1D_8)oPyep7P{)Ay2VyCN;wvX2~KEwT{Wf6z-{u7L#5;V}AULZ}~m7ce4An1AR#Rv`dO$NgHuZK!$aGuWy;SMzt23viYGy8o`=7;-c~cNGx=+!!MGJ^jYlA#!_iw9{*FmS7TmyRf~4AfseYbDQ%(;Z@<^AgN#1#&t=~0_$xf=VnepVhWl@SD|?AdW|AMf_I|H>u+K-F&e<|k6Bn44r*)+Kp^7@1jIESqrHGukvb85L2g5K8451d5#7udxy>jZjv4S=*k!g`n?-#J!==MEC5Y6v6R38OAY~(fu8KTgu!Iu!Eu)^DDQbqw_kde}02#^MfPT)5hl8E8!SrL<05~DH*h2T<>z{>qs1Ep4azuXi@(2T*IKWTvSO{XMHjEKJuX?+~h-ZtZr@N9suP%1DQTYvJXjEB0ifakj0Rv|xBFAlJ6_d<6)Z+)=>&cX_Ur11eH1Y9jt;zAHUq&*D-APZqSqgKpC71liVJPWprF~EG!w<5K|cOGMGKLw6{&-I3ml&COt@rm?gJb=804S+4|wa&a)hDEl_idDsoCQA;PK~hcu0+Q~6M#;x-L1LI`Yk?k;Ijp5zseM6uMuwXfKT@w-R&1X~fH7M2f_(6xan}3pU{=v~qXAS_*0w&#$uP?z0EI6iHfzxrAO<{d_-`VBbp^&HTiDouUm`ic-|aG3O51>KJm7K5_3tFeQ+A_E|6E_mvI+2-uoVQr`e6y)Zgi~hm32LT!0TkZUJII$c3??z#TdS@e6j5)J04gW`j-{YX%fluL?X_r!*emH(ZqYk`Xr?`MHAp$v?Vy|#)!A<O?LbvnK<1HPAQz0d=8L8NjYke^%V&Dh{`R;WBvC7&Cy=6YTvC3A?&7R2*POgo}f!$=YQwmpZUh`J63|0z{X{Df`uuP;GOPfRVP_gvE<K_+h(Rkn>PH1*b1;>MQ=6+Qfc-_LTF<l!DgKdihOGDI=ex_Om@Iz=-~jInj@%;p2ULBgMEk3VeLj@x@vpgEj>D=8#IB)zp`c|)*M&gPJE7M=jWn|d3K+Xfe?R8ji|;(Go&^ht0oU%lWLm)y-R(Ggd7IDnTcw^YHPkHD&`#V+YwuqOR2PZrvJ@k8kW3YrG`h*T!e&~7@3-DH;nW%yNLc}H^m#a%g)zKmBZJW{W_Sg`VB<o13lkZ3rm89(BHU?0()a0CSx%vtx+hh&!`B*oc#hFkXr$o1_ev94bt32`3?eC;J3*bJH3GjiqP!IQ{8XO{Y0NRN>?@Js4Crkc-UK}7W&tYpxEhnrqd9@@;I^b<yrmG2f0{;nyfnCF&*u`^bI*j@}YZsz}vF`B4o=o9%`6V+_hmS#ua?2;6tA3Y~76eqF>WV++7%_;xTS}*qc&hy^z*h_d}H5ce13tn~I>k<|O&;zZ4o0-~IYVqpC^aw`M{lH!w;4gSJs!-RXx4$qT~Fn7FTJd2?S2hUo%)fG_{KNA+C8=o~$5$1~&zC2Y#BSg-Rhl@GFj^-0pryS_urn$D(OKyN6B0g6m6!tXq~S|`O7dFw5NEL$@i3Kz@Ti7KTT;>Q$wnVA6cm1KRq!Q3D$>`jvO6OsHt**pxKBFLkd-MTr}fxiq}0CU69Uf#s+@1s;iy6Kqwr1D)0Iw1*}!&Os%XvYD2zb+4&)lqjcveI>?rqevK1XLGvy5?*}(P;!dzYEo$B56h*fZZ!z>BiI^M#GKU$p+~Aut+tg@)&svi+oSW3?Gbq0q74PgbTevsEc_{`%U@Rw6*bJCLW&9#w3^Xj11vv9S5D;43ZKC2Y0S}d>mo{tIj?~-KfEj>Q?82C}zI{hGl|C4%ZKN3Fx#N{x6f^{IAwSTqPeeAS+iq%YYTXel#AmlQ`^DWqNn5HgS_X!U07=u7UEox%#tg=Q81yK`k8^K|;Wvy~uxOv1n%B7QZ}jHN_4C38?M-D952$@?CR$;jf^t*lnsc!^?krbzVJ#r^eucv(ety>mD_p#*qoBpJ}(cA6{Tq&o~pEIFw#kysbHwkZ`n7z!tn&4xnn;5yY+Og_jTC9lrK80N)>EE7=xvkbD|9F$f0Zfx`WVCDTuhbxswb2eNDoZNKX7v>BUV9E`Qz{h%Fu4xjS*_O0~&k*h*RY)QqLzCESAohF$0J(uEP20$xa)uqF8;OEOS-TAvwoIyU+EEI&l&06nMSR$2{FndU-RD^@B(9sHqMp&0)J59-D$r%(x-g3~1{Ir}XgC@wRqwCOVj>wW(#~eBD&1DVt{>7*{@)xjZ1e^TbK}<r-+~3=WGM}d|g|GewQ78R1so;JiM=oX7dGU^GpV6%G4({;jAdTZEAQ`_ueMHG?^i`Z*I=xUi)rX@ZRH&N3^hc<gvDXUC*0Al6dC5}><4SeYk<B=kHG)f<8Npy$fY{$`q^Z4qqacfayIJ#)88a;WQ-T~N1tWZ6e1OQ5ndNh!0tN`+SgQ(R5eSO~lr%<4qR%vgnaG)zNI?wVXKix?%!5IqD%vivr2cT<dwFDpMi~8!nVfFKrj+GY?<d1%)1{eL5<sGlALRxvBpb}ecjBg5dY6c>_^I;Q7T@mJg}VX^CL$&dNRbv=!Q#U8YbldVbBN;6o9j42weI;)nM=ldgR@q8igy^PPYE1x9m?QbxlGKgnV<O$lzVSv@pj{KCQ19&^?b<>X1-35spNLwU$I&O(Ye5gCf*ji6P+_#c%Z?0%zF5elfY8Xtz#+%_HZU>R||Pf@REGT5b9fN!<}8$>~EH)_1xpc$}Zm|mJk`ln0=pT@N)>s_~Fwf+xuI*H$Z`9V%<gAJwJ5jew&3EWc1SEz`qBmmL->K*bZ6w<_j}V&7xWSJZUhLoD(dUBUdmRq`Xl$bxXJ=%*Vu8QK9Li$4(D*Y%LndEz_fu6?~3kP%1&ur!U`G>=Ga0C*BH5yWA)#1v1mfmZ3uzkKww6W;~XU=59#|&aP5tL+mcxCbWgR<0qe-#n-xR$lq;xS2y8DbT9l8j&8=z@55eey0P~$U#LW9N|tLv%gD{4n{uqXXiu2EGqv4Pi#aazeHhdUJWz}HG#p;5-eV&GKPHb7gNm6%{P?A+B}8^L8FgSQ7Jcpg#9y?T$HF(k&?XVM4}cD+owv2djl5{q4M9XvzHbG_l%BXCxJO`n5<rx42*$kctp=qCl3{cpOajtY1n_NY$tgCI(WIVG%N&f|F_9ugaJDG@ouXpm-Wd%3@!gL+L(rZxZ9Cs<mMC@3^PtUB;j$lHXPvMU&DnY$mrk^{Hh<fQXr;_ZcJJ-$ZNsTIzEavORb};o_31N+-0;it>CHYO50BqjFJ&X(kf*)Z8Tj!v-ENCTPq@-0$%iDTI8F`})QJhs^S7)Wlm@<QL53-0c=NKRRy>91wLn5P<SK6rbgqm`?ekK<D#k(brw5vFtSfO2V6vHYwa&Ct5Zi5hZ{3NBZdAh?J(1pa<PQC}aYDuCd_><7vE4Z!WGnn5x30pvBkQVAQ%S(f*Yau0N5K!vWq*!Wcu(GKLIfx0-kpAA!rwyw&O@nf9EO!akA7RNeo+=(D>yZFawX%kTRwFCrG@CztBZPEu)D^jhJWknyO_itLo@un^tjoN^n0isTzMpN3J`(1%BJEjiO90F)%qt3*mwb>P`jg97UTmND*vc0=hIYyoCSKDuJ9Tj-ESG3E~jzh%AM<#L>5M!v28zPHM6m-MrsPueZgxq<hNnh7^dc+XOUr}I8(UKULxe`E%*DJvUVlUIPJhD*Fg7$D0bcaRj#1IekqEA!;LIV^-XhG;e+4}uD0wW#ZZt4cIXR07WPqqo1KzY0*P?X)W#ICp2u-j@3Nhi5Lu;iHGRS|*`UHys_o1RTj)v7m3EI!5F+I+7#DKVj_;a{VW14R=RTG8?i;uZ3PJIT2P5HPAH2o1LQe5&GyW*ZnJzif+mp_VvptgD)2`~ze`|4<<`IcYXOrN05t`&uyB9$N|Kkib+gU=}Ucm{M)*JPSCFIgI;xla=w2eI^nGY6$A@ranl*JH-%Ees@SBL;+eA%jR7+#L8$XCiLtWUNh#cO2p72fcqacfZFRQGKIzHLjG+~))ZN=2<i>EcDk*F%U=-paKl&MmZ=PWE2w#P6p;EZL}Qz)>6EY*|W>rnsEN8_Qtj1PbRnz)@i^XO-|rB=Qa|RKIFP#LcCjZsxOlG~T|8KV(^r&&nR(s5#mbnrn%J)=`Na(90+WAg6%3wE_3Oq?F9uxM1-Bs@gFEWJ>D1hkXZ+P2fce)>eqDFOJ4n<vkWwd!7zZW&?MCak3zeLillTy48H}X~5LqcbyZFk1|s?^uYWoF5X#gH)rOrzGmW1xz#>RR`k{0mHV}HEUG!y@||ilxKI%J;`SB4lp%>)(Nc3R1XZw|h@v%lH#+H^j4DzY?tQIB;AWyAuBNE|R^f^A6&XiNw|BtQiqf|=zMI9?ie!#ciD4o`jf(CLW8A0n>g!Fw@4`C2v?5B#46`w8$wZIyAvz_w2*Sx@{hJnxUX5OYNSqXgqmMp6yEg5eNrdy%dEWv_el)7Nro4+zoMc>(-$vMDhN2H~9e$ETT3M6P&HGDV!xjf6tumw7*qzso;69Em0`a%~T@?OP7kpBC>sLjMtA~TNyzoSttrYC#25>)bSLMC+6J%W_QlMQkgPoxYzin)b+6j5i+^k?o`IrMry!0Tl{aNeS-eY>qd8}JHLf+2=Wt+bgi`yoi+V$$LO)s0ZI%d@giH6cM&jZ_{$(&OMGu=tn#DCl{;DpalQh#EV2qlCnrG!lVuJgNBM&6zjW%QLP1JJV1Ypy?FdHkWz$#Tw37VF9Cy4oyEoj2B$_RkNEChm*omC{dVO69IO-wa!yW3B{f_dPvCq1{6W)oRN0bM~2f8jaIJn6Rc39>Jax{A`!>+6f4L-x`IrtLqW)m7o$5<2}_PpBX%R`XB~8jbHx*DzCUg*dC-m#k%|6{*ir^!J!=h@igXLHCuyPub6R*py3XK$FmOOyXEF0L@hDr#{QqH^VV(<37}}dKe7b{AaY60IjbWlkuzVvxZT!1mL+E9y?YT1{Y&0)(oHl~Zk*BsNnz$hA-l!y4|`zo+yKag)IQpmcB=4(T1TE>Zw|7dEThLG!PkmeK|(QjLEp8^&XA1~l3wIOMM1(Vcj^+-mamAI)#=})LT~K@o0|&V$m|aD(z5gikT>@Ge3Hq);?$R0f!8eCDcOQ8_HT`R)?$P^uAY7d<~yPhU%COEiQJADM>N%)cj?zE^s|hI?EG<Gb_pxCG#@v4-b^RFCkus0#CqnM0W_#4PuVm9CE+!k&DIm8=j4C6Vyn9oY^i6)Yj}_5<QoOFz#EB18}mGx6a=Xa#LU19{7Gs>yuoQFl8c;fWrdl(rY`gx6lJGcex!ARr#@PiM84L?OHqHIhYO|_zr+ui_inAKcRCk#M)=R?7ePC-a35U*b+Pj2idP|EDmu!4d34jTnHHkQ$AR7T&rsXVnlgS6{Y_#Z6leVJI$E{kyzRsrs|qcQ*2t?FJS3-y6IUtG1_0iEY{iyz559JlZPO-Y5w!avYi!<vU0ZM(?E|I&U5VF5AhxXF=x!vY^;#am86T^Dd&>n(^gy^YD~!lx2O^KyLgrtR@9fmJ?ozD0<t$v`g=gAl6pBWda=YxcWAr7fI&Ik-UR*}0-TY%*fT2L)rPB$QuT5VSBm;eJLuy>>KrSe1$!VLiK}7j`8+GrdaDLO@r)5HwxDn`KW3-(`OD#aY`U2`DFrn>y^^dKs-UV`50<@OViT)#eboL|uee;{N<r_3+0$4Eq<QaQ#7z4x6laape6j{??IsN<Kl(DY}_!XN=0=E$ewR%#Bh{H!B*nIruKeGIoEdU*03SJ<JF9wJ~<;TeJ=|Y@x^Y`yd>bus)B*L&evW?Q(6YnHzPe6#6k8cJ1$NF905f(5N6h4;8RpzirpD<_mVj%n1dfCWkeu2igCa?4{yA86P_S~68`8&EO{l3SZg{QymLWBf7HR}4H#LkrOn{d!`B3Ao+a{pKD^&!|PFXVQ-Mp<0!Z8Ysf!EsBI2p%{RX5IgaLfIdo!M+IhgqQ*P&MlD3lG4Q<`GqgUMpOzpSE!)TvG*qr(=SoWwF%^qC`gJr!tKcl!H(|9799!gn{!SgiH$gSISqR6RSJTtEzi;JcTNdXXvHJ#=E}XOUj76Q#m~h68cS<@zoj~SK7&ke1A~Xk3P6NgIgLW6b!5B(>$GUvE0tB4qcMAZ*I^h>&qCDR#VFQT=+T^$PYaBR^=VMeh{pM^K!cMD!>3pAAHWI3>|Qi<ML>CicRZP~f6rfzF43H<Yb05IbZp~`R%G_c+uyBm>+cP@Ml&rc0;NAgJ?>BragtH()eUI3zfzUpKkUrpCI-`+p5HU}!3i?uR_TKYj*R!gQ_Yj&FsuNXQ;&bOK$*{NAK14XC7V#`2H@#>OnyA9wYi{^IM7~RAFk{MOlOdjBdCUq8d1N~PJcQM>2t*I_RnAmdSf;ZAY9aE1Ef~l5Qq|vn}9<?{HCbhqquB?rxf3vF_$TS@^FPGr2aeIn{9AT?GHXs3KCEeT%OXACtwtVUNP!n@ST#CQj3$f?9BeoTqehJ%j<Hvem*!7BYyP=m#OFi5t?LBp5bw<V=5s@9%`CvoXsLL^qeEORONQin^?5CIg$kRHxvN;lZps}f9MaU2NG@4@b~h;-W|pB*Zx#p6GOlF(fLG_nva;*rQ2>jsEMllC{vOIh3o`uWpe#v0@Vqso=ogY^zHlITfpDXjjbARXu&xnaE4ctiGN+;eElo?{9H=}%<u+&j#EbYch$_l=Mbd=@?a5`ES)G!<u`^Y8amb{2wc<D9bg{0zOH7q2a8Ox4h_RTN1T4cwq*A;2p10jU4TDp;Z7!RWffdDPR5~t@)C--><GNAIP7oPsLXsVs{H^fV@uyh2fRkh1(N7psUhzpGuz+>6UvEHmW%8!eIj7iE<p<`UYO#orY2}Xu>LTrDQS2?Olx04g>h8}gqmUXNh=n0lzQ+K;ESZp1%bZfuQb148`&e=FCd)*VX=_$wt{Lfk%KSWs<-x}WkDm7Lm*?AgLLKrGO$Stof;|y(goQV6^LGVoXPKbE(17!&E?fzqqA{NYC4#1y(Z#mgtPM;Gb;AFt{7+7#o*b`k-%mu2h)`O-@fjCJ7bZA%fTx$pf9-7(|lz>=l!*k{OLHiM_>7GIaxs%zC+!eLa3d~sj3F1`m)*}H841E4ba|B>>Ozt;2W=BZ!ZDn(L{tetmY<kA#2{@&6H*bTE;)1g#Bp<3U?R46mxghz`}>8?4pAGok;3#O;ar}@I(8~Y#B@-jcO6AUs~OdV3KAq{$X$RTdyc~w|VUn8HvOLS0cPA=+t;QOaG+44DFvH1)iYRF)2-(J#<e+Ouu~nPW*&k!bJED!hg)TW70!zb54Oe*3ds(8qH{pe~zo*n1nzDY)h<g(;Xp3ONl$s3Woe>CMlQnnQuY?<C+!s_(6ejnZR_MTv`t=-`f3g9Taxs$nA&D>hpcuZ?K|{kFg(rf3C#L&%ttbC6F`fE!6R9cOY_#Y8$XpAfiA{Rrds4_i;7h*r&4O9tyUOiZNbaR3Ki}IQ=!4m!srU*iI-d@2)7`upF7wTjdHwFKXZ1WaYbYZ26B%g_2mh>?8*g`5}IB0!=tQVonrVH0}N(J`_IwYrbV9wZum|pe}%;{b13kvW^%SQk)ZiBtz7xx2^sJPu>2l-iZme$_dFl#rNSupL`pBnMSGlIAn23nvt9Zx`GVtjXyBKEvBmMFwdpyxMOF;ZZG#u-rLeq<AE$1T?RXwr78<{ka^SMbjyGL;doaTjsIQa6O{ROqh#~N+D}Q>dM+iH%X_!ibyJGsbmy+0%j*m!D)2L9<gUA2+?H9RdVv|AmQf|k9m_`#b&7a!%B!bbkB~ssq1Eo5aBb_}Bc}{9NP}fb+Q@Sh`Epo47^I+j7hB=ABvY{VqDU<`KQZd7L)4sq*@Fa{Ozzm@#(v=Bgx1iDK*_UjBNY)Z3m-F3auo6{*7B0m6_PL16V}ISS}D6yz<C0M2dN>5LRaRO+(}L!+z-Y5VqA(@xc(_N_$QHsc=?0Qu3nv*LGkEg%faX0^OM;aj4b@)Z$rAHQN04*hD@mEXu73aRHl^Zotv|HP$-}DNbf!08EZgNEXknG#0Mq0XM*vWG=Dur{DK~Kx9MS``5jCs_X+L&9c*-bw9GQvM#aYf=~c(KH(5@3FWvs>Bf_0Yq<HXcUouP07C;N~<Vj12W?$u`!pgGPvT?p0`!x+7_CdN@26v48XPE8|$O_NXG8IBku*lop6tdSWP!+kfu=P;4%(6X#aI%E?yD(Aq&c1Y-3U>-wZ8WH}?A7dX4w4c^mFmp~qld}nM@F#)8g_|oFfel2Ua1}s{p|w%PZZHNXra`!tT;-NkR3^B#$_TVL4lk|09x7h^FvU1B1C1pz(A9IjI=<~0mF}f{37gWpX%u{iXsX_*PiM&r3|GX4!7(h-YH#|s6gU_;3QgnHAT%~)TZYLX!3{hPEC0vLT^@pd!WHMAYk6Xndgfzqb8_vBr4q<%$*&X^~2!o=8#RaY9GSM%m|OsDz0Fsy=QOInA||=oNkECco=EltzkK}Hm#MmB}a@L&83sM+bl<Zl#Cdx;=1`|reXCC>EU)+=)mpJ<f>#+j}{&?v}xf>Tbou;St+hr0baOP8A&uBE2EK$kU1L6zk*b8M)jp)7=uzFxj`N$p7i2H1tmq_<RBx~tFF=2AB&U5j(rWXY%AlTx@<fyU!<@6UOU-0n^o%+-<C3aJk>8aoHe;{hOB@kK9Z9;wLsN>5s$^NVJTVqMvax~o8@J)P$p%+5OVrDv5xlaS~3wvo)nsm|2BzIPtwv~l0bsazbGC$r-F_MHC6^d1s%tNm7}Hi@BEq&7{9)JxHQMEq0+>!&Q&i#^bRM{0tZ^(hY3?c{s25O90|Eak71>QW4&!5*`F`QGm;rOcO32B&AtxlKU&8MC%vjSBLR}TVR-tX`qt1%pF30DBhjKi;PCm?5q{_Gx9Dm%C?vF+G+x|Si9o-fcW7t97L~I?=vm`>QAWhPAuIw|qw(0$EpQOjI5_VMqaQ!V^~*pVgefXeo}E2*c3e%q8z|D-L|knZ*~R4&>seb2?McdLrYMueebDW{=a32s7>D()M})i4+8#zKr@PomS4(o?uZ0XIN3FS*0*lRJ10SvO@8K<HdFOG|1<@JZvV3seNIR0t-r6L2wkXK7N2MSMxS&KDqykFC9H2#qvC2fVzy5Xa#8m8ZshwiY-05=`0iU=<_d!nYdIbH^SS7)fVe%4E4&oxRq6W_ZvNk6nLJWseIWK|OFq%^?9zP>ZT9=-!VHaHNoGbV0<&NB2Eu}GC-!hr49CIaC563c9*le~=F_d^+p?!Ax3-+ulF{fUi;i>7=TZ~_>mbI#LEtCsRA(hmFrE}~9TS(bz`We`o)pHDXJ(-)UxxRA(%#=C48X-)P)VHxnn3l8Y;DCUFZ9bgpaFDr={ez2>dl^?@qu6FSM^-Y~O(M6xi$>606lr5SpolDq0wRdE9%j+eD_cnT@pKF*Y(_*eq4mmH=l&>|w1MD^6;#CWu&|k125b#CMu04Wa3a!LuVrWqLi>20o?l*#|L%iXz^m8?QnqC6!4i`>LhrMqQ>khMGXVyelE=%bis|8$Fwz}BLpurYAZtZS{(-D61UF$dH2lXBAFdq?j4-%u1c?AI!(T!5pb7p=sb4?b!Cv|NAwWQBVchc2AOI;@PmRw=l`0mP)N>d+T_OR#K@oB7`Ed{{Xlhk55bFTLIZ{zbMt;-7F?UBY(d@90=_-pHpx)t5qzb<?Kk;&8+bjvI;AC^46|mGZ&llk5k{`|(DciKg#&&~1VNmo|8m7IHwJ~p+6>bBPw5Hltt(baL?43ZE71|54r905N``w?A_OOuq8OlIwJgrRY1tk}1f)=SA8CnR2d`?xRogdAu8aoX7cPh%|*;TRw9X_vD%_cH@IWq?<+FVOeX{cJ!n^Lw%PyP<fxZ8@B4lL1?c$hm@o8q|{rmys!#$S1C0MUL&Y@u&}V(b$)zs(Fr{h)i`s{L)eKRiiD-jp+!(U3tVwjz7ZO|O6^2~r}T%_v7gD+BaZu0+bL&ktW9pxyGT+?u%Ie`^>k9}01c(9e{Q+LedL8m{F2lH`owSjS*rXhFP|@Y07A3P-Q<6X6E`=(sDK2WYY%u|wrY?jM*D+A5BRuyZ<^m~^^SIKQC-tJLZKbOMWZA{pR$o=Hc0vL9IxXnI(y1nx6k^#bFH%^66txyu5FIsDA@BRdoZ5RCaR%JWpO^A2jdeb?rE05SF7*L1>ow3>|z5Nvm%4E@+h{Z&)=TV*+c>8g6=Gp0YuZ?Mx%N@Vk}G#H4Y*&>VDS6g^LSrXN=c?eR%{i+PcNLG3MB*^hMC}-o4I<ieK+wECOV6!TNyJ{%e;PCI7CaPHSVJ6nKjJ~YA#5k5MZ8@^5u!fE}-XRoLaA7aUSEhx429%X#<OB&ZrrL-?qQHXUNpU^uVuS*Vs+Ub_KNtecFyQyG`1T6oP8vJDiJ9nfU)l<hc-b`4Pt;s=p$mC)KL*;Woc<Pth8W;AknOg8Dsu#GMy*cv4Mp88Z25^1TYHQ$p$GpMuZFR!uWNw5@kPs~V(APKmb3cGD?9vrl~LygP<h?{x>z6|nr`ow!YQ<}t{W=?UfNCkTRVvSGxmE*rBfWYzIdkGGopGNqL;5<Mc0Y9-+K@@FN6b&bjXq#UwrZb*7^Bt$cKW9j1u-ODGaLTJ??jCzc&6JDV=}cm?%T?>s0+q_DxYj_i`=hM2ijoFc1Xly&PlgNlw}FBWeC!e54tR&JNI&%l0-9v7QQ4nC!QZ$H?Rvmk!OmFfdsAM_K6V9+FI=eOY<wK+2o06nBlRS!b_Ieb_~k@<t|-bDZ@{elTTed9bQNC!>&ojY8;7V#({Q!?%P$2Vzq}Ii+|!3_H(@@^q}_vy(O;d#jD@5df9jFwgTR$-ykBX~Lu9umE=t$US)5-~fTJ2~VDSAprEl(KwizK+8UMwi(_Lzi>Uw%P01#+fI!zO9-!w5y0?!0zTx<{vD(2VPvxk5ACQl68mc#dzSRRt*o2n#G`7TJQe=Ku{woUsxd+CN>4YRa0~Lo9eCnPTSl>%A%CaoCTAL)>(}%BBotcm_%zeUo_WroxH|Q9{TI?4w-%h!5o|#RUx}7#s^pLvIbqZBK3bAM$mOZm7is6RSG~%xaDbwcwfz7B#&sGY-5QD&G~RM%Q2!LqJ8yCo^&bkeBL`G~a4m2P?O^Tl;75vu9z8#29;pI|gFD2`OzM8Gh}dbnIoCI%QCK_P0idI9Sbxg4aIZ6yM$7qk+8t`;rtAQ1cXSgrp?gq#7*naIp|2|Tlgaf;w~j)neBzh|ms9EZ*^~Nj4?aO*$d$Na8t0+#GjzXUO(|NJ54y%lfLWh48B9L8^W?f|8Y)ojRfR7>f7ezxExDw3q7+#BjyzDr%jw6xKU}`|4>EsQ?K|m4XV|EtcMF-aQp<xr1PFtzFm`&mTF@#Jsr6B4AqN0cpnJh2dfAdfZMa`5CWGNPjG8ShkW~L(s1^%(HbUR%8HFV2(5`+%&r~DL)QPCsIi3dM$0-G-ZM;HWOw|U1w6pJdCl~9XRqOc$PXbk5k3?yCf=!9|f#o|%MZaA^1<woZZwWm0VY~1$SOqryU{`KLpFwsS&^WVLKUN1j<ob%V<3c_i_q*l3BE|j`<2`z@;N4hXQyTdq4<o_tP)nCYgTzLfny9X0jEQvL;lJk;^a=g$f&5iNXT8rBYn{Y&b@?e#1CYLHiqV1QddiHcUXGe~>r7*R1Ki&}SZEp52D;2=sHIAks*rg7ougL8k=Nrh_~CYuGIxWNbo-0iW}-0zWf6!GbBbomz^&%?{@S|Cc{I>vo_-!51$s$|Y`Y@xR)|e@TP#ic8Is*{tTBCktoag`!*x~03=5d7@?!LQ%@kHke-|lQ$61Z4(4`n1#ZBAAjjS~!IdCnGY~SuRyt%oR06_h3$Tz25Z*sBc)i13aHB_=sL)vr0eA=xN3rcJhH?lO6$vURJlYY3@7V3nl&l6dd&QqQeZICt*^<P``+n={C2h>5i%1wQIk!LG+3hsB;OFACk97{64kUprNWBB?O+p-E@u|Hvt75)kk&WjJ378!?l8Nhj=XG9?zM)hyARO#&0=d&%}MZ5UG{eU#VhvrkC<#r{pgyL}&x9~*LV*G23c!ot6+MOZ~P;x&&3SN%~ZvX;{g#=4dh|!r9_34j`W;#oQOnq?l);zZu_sK^A8<|e~-M$tY5?t!fZN!d;2=Eftba*>(Ywmb4qFr#Bhm;$=RCxDOG^5?~Z`&Rp?4dRJQr|)cnBe7ndXGhRnoW3mLfr9SOA1%;Eg%+cy==xtqUK-gHaQalf6F5?hMXFn><Jvh_1x{SXmOPx_M}}P@F)aISD@sMkuYCW5x$H`kG0@Tu*COOV-SGsk1=#sFEn`AJCf`a=|7~YQDbb03sday`i*|%RjQ)^B{A0N=}p-Aj{e{W=Pa@GO0T|^yqyH>b2s^IoQ}~}{ax4Z47at^okBVk*z<2gw)4$b^hif3Hw~DSBn0UF-O)AJe6o6oe%A9GgzANTq@}h)do0#x6cleiMK1Bl3;;=+VUBAN0dj9E?_jz3bu4RTh}#C(hPGFl(-u);g<4!vhRe=mb*B4cE-yKH-cx)F$Q|@4Ty_c@Zsx6?@D}Z<YilZG3&At5p?M)%=!}VyZJMiVoX0EN+|Y~I9|LD+$*}eb0#3Vo%wEVvU$4AQWR;b$P1z!vRtCYc*vfSqcD*rOr#ARN62v7eppUsSo<}6(xJim1ej5y3N=G;m`dwuFZ+VGuy9G#JccTv@Z20}{DJdWMrP$+Tr^K_-wDVyroEHSiyZM$+lzSje&_$|U;pMx2Xmav&m)sbzsz>fPqSrJQ(H}fHw7o*#`B<vSOyd~HYO0rM{mOZz$<9kY)$P8h%H*jKb4&zpij?vcyP-0SRrz#Dnm?sjf}=1b&sN{lZ^;CtzBx1$0_QW?UBdONaPQhOon2%ZkNi_vg!5rXP-+D_uh!u<e~4R<PrlA0>lNIu&`EQ1;JC)4cVq4K3i=jxX^E!7NnDf0=^_m>$iSPPF^~HWkw{A4)>V)#JXZyES^?`YM@nZ?5N=PMQsF8iy7>avuo9g?&@m0l6B?*{>aP!6>y=_q42BVQ;&C_Xq)bcl`vVnkXiyng96@R*b#p{d@{5Y9y+-QL#VlDSmUJ41isfiOGf1vsl8ow#VB)m<B;T68@AtobpH{yp<9j``ylU(A#I<Q3T#?wiis6heE=S%{JPYO#7G{BG+1un3r9EhIu{hye6`&BKX7Zz#(Q06nM|x~g5eKpvB9NJ?P>`xIp^#m4!d_VXz{K(VaXRCC-2xrk=JU{&37HnftS>N!jOZkCcg8&7j;deSJT@Xw|6^9Bl2R&cKhHo<Ogl{b?E_At!cB+Eva-H8W12z83n7fY6#?I$_#CbsT?=OW@W|GMy5~42Z8%C!#VxC8`_G3a1v1PyG9QTkwg6|w)8k_(Q_b$2Q3-e?M{>{KD*h_A%o{n7?ZKqNcFWfEVFqnb{XL4#k1EkWR^77H?GnAz-uP~BVxi@Wj!Av=UI?yyW4Ztnp#VZxRMblo&A}0(SIcxMSe`?8_CC^3bB*`(_}ByBF^Lj(EO0y8vlqx#vL#4U44+SJqvcPf4Y-W<Cb$K@^clyW%Nv9kc%vjT?)^5rQShO|QXcUJrETK+&G2KqCbV$MFix#b3<8PU_}f*=>YYN0KCH?{7(l_vC9K9ktV%Ujnb}CY7@cr8+5A8wYC#mW%?uJpJompKFnBR=K0W%gedgggpqNuFi>Z|(&F=)rlaZNC2YtAZmIxa1ECRhEM1g|d$gt~otEpz4_(9ZE;og&n%lw@c(rK~CN1Bo70709kHx|o5_DJ@qo-p!GVcan<G?YaM6utxwnnV2~Ws7@z;W|_5O_`5wio~@VOX}Ch&Cu;<&JZQ6H{R~ALpwiT%Bt~)z+RH7Ln<c<(#;G(4&{na0-MEbF=4xsZkmWw!(Q_27PM1uKET-`w7i$mvH!ixUaD*lg8P|IvPdMaMl4btjn`ck;Koyh{TsoYy2gY+;nEXJ?NY;2OzyiXm3cu5@?MUsnZF-hvV{6FbZNd$l-50w!TnE@d4NTXN#R0d#adR~|LEb;%ecJpQwTpJZNKXzoP>Kc@p6zE<vN}5Cpy*3?u}Sg*i;m%M*Z$Rej5wUYq=?V*7q#sTv3535`*a)h~y#B%CyaemV26p@$XI;9ea5<oAe*8C}dariCj<XIO;!Duhen)NJFS9$-Q#df_uSs@CAh2t=?ytWeJI5sEe4PRT^se7*bo)|Jc9tuxs+&UN*wQmP9lU!gl#{B38KeWB~=6#I;*lg=3U3vMNi^j6>?htpkVmy-5$!==v26u>3&(YVbGPb#IKlZJV@J<jI<sUxjoP*d)}V+m6_V81?hjyq~Ni;_c0tT}+WQYcAETs!0A_8jsX)vil3}U+c7(ZQzjZ8Sa%65JD;~?;7g3pAQU-7saZMc1q-GD8oN(w-r5Y5ji5xtGVU@hI?~o{!m|UzG^*Nw`3;L8mMs4U+kq124FTBR1SzH$@f~qFNUcd%~DS%0csE3dZ3=Oe_OI-*a0V;6UtC>ximP?x4e2cSFG7DqSryrGLmKuUZj0g2{W~7Y9NBha$QD>l+J2SyrmJ;ak$zt4}(^c3X|hoS5()`8n8MNPbxBI5XDD1RRaY9>|LbXc*J<#$TG2DAAS@z2-G(o42R-(e%GzSs#E$dFv@($R>sCuwIyJJO&0iw28r8!XB6Z!;ayERFZ>#OridBc8jP|(@LVtFN})C3s$v?yem}sU6Mxt`?L665Le#}$TPc||_ukYZs`V0jF8nEoNw6s^+!Iy}+@fIczEJ{nynu_G1UXZE%5#Ooi|WA%j6*YV%xq;XWIG0UcBA~3D&vNQpfX-=jSOBHtsuU;h7%<Q3cxxKALvWIF3#L&ME(q(%Q(<|t$Ugz>zl#m6-?m1sn`hI1qYw`&FLQRxn@I+W{W!lh#IEOD$2LNQC2n&On&A`+)%ny#SpFO&x#4KdgaV?Y4NWa2nS^S4r0A`$wp}~-%rKqC^7p1ovHCtjF_C}r!<!;`?a`^KO>Rj7&_eV;YN~@NzC6C7z^Yw+6^}wmMc>|&RlPO#<RKP_DB%*M`VZ{k7#oed``k5cUkhZq8$?-p>WU`&pCs9Db77wSv!`$yDO*eg9Prx(eHXH6Z>ufX!-k=D66%Mwk!Q{-uyu~dC2+SelPU%vYS;TZRbki6)piSqQX)Qy>*=Ui#EVfC1ZfIiA{B6F8Fu4Nz4YF*ae6s{<zRP<%UEzzPJ%_JP9)5>=Xm9CxBe4Zw3VBNCg5(!8OUYlnhGvZgG!9kzl|%DhRpbr$jQZD($Q+{M>K)of#oUJ_}$(<(MR%K4?XUVrqC`IN5NOS8f}a)f$^>q>Ubvw$14rZ8HRgPCXOO0)nmc(Nv4@<)G{W2so&Ba03A}*7ae(N~kS!iPIIXiRi+j{Khq|T)zD{Xx;C+uoslrsxLa#4hn(tLu`f(&c7*A+k8kf`b}oFgXZm`yVhE04j@Hm&ErM6?(a~g3eoLeQ~}oRH?+GW`!rNL6Pwwt%u*Ha<*U6_lHUcL)R4a<V}Gf%D(crj>QrI-HW=u|hOz*yIq;!UyzYgE>Dg)7huoZvia=)aZ^wO}yzol6Zes|A#7hHJVf?6Y;A0K(dFIU;s9hA0>g2SE@>_JD-K8Hm0k_Wt?_JW!o2J$cH7mx58|nt7>{ip}%{Mu`<uph#>CmFh)s>B|^s|sxneX1!`g6Y9nPw_zH%ODvKS0y<*!hVbnqm=LQT!vaFmT@V<w*<B_OzMbusl=xI5mUvDTcFV;YSO&UkR=hNSE~lg9-7>0q8NTNQikt8wd|KMGrMTFRlt3R}T?8yiqwJd*Q%k^p|QV+fO`!hftlwCiR5KinJ%JU&RG${PLqjVph?8$^7fVZ}^gETbhwk%-S5t@qDi)+YCEMwqP(A5+3pSdfBLCvvUq=C{?rc^sb*_Xgc3Bl8?Xb6`#lhJz!tBf?}Y|l)KhGQtn`og!-NbGqy@MMa$Bq?G|j^ANzjiTdDYJZ`KXN@gVSe#1~`F@0aK*GdZye;ml0wI^q`S@!Bp1uK1;}`=*-=R$(pgE}ANA7M=!OG^p$!gr_vrWd_HE7yWkhO3ufwu&r|oE2TmFgIe$fikqxTWHE4sHh;@KMd2=jE}CiQnl;cio#Wa`$zh65KjZ6Z$Q&j;K?G|KmL(2ZQ<~Ns^DU%aAr5ZBAMbflB=}RM%?vL+yBC7KtJ%g`eS>EA>n{OyrLXTM0M&Xi-eRTm%n-ybv7_bI#|uFE{8+K&0G?KZp}Peo_hYS0Ce?EvQOS4Pl{*DLEots0C_GK|qm`F0e8FOJUd4$|Wfwgs-CVT|_w3~fQlhi>@$~8^A-ITz)}Z<dj!Nh1rwtMUN>h)?Oa`;?vbouyQk!+J=yzp8NdFxPws>NfSlK42pvn5up;NcJAj9{vXB40<D7<{qfZv+7`{Bt`=-X#NRX=86p?7ZRjC}2v+EMncdvflc!bLD>Fefy9l5Gm@U1$}k?Wef`3|E}BzRBVVCy)-}!|;NFdye>CYv=96Qf10}U()nC##xn?QwV=wHG;O5yJ)-I%$%+rCT@S3(|}lk4X2ljEwhdKiskOoP(n}mFHSgDCR0e5u<}ord6SY4n2m~z`uTr5hO-{`E2=cG-jB4=90x{ep-GRx)p6Q%3NQwGi|IXY4dO4JFY<ZEa9Z5e=9+6+2Hdcd;TPM#CDjVS6h<*32=CIZ>aEzVDBp&miacT^s3a>z(VPP&76ORLfI|J=Q^SsWW>X<0gnHM@I_HK#M6`Zi?Z=ST@OQoB{^;8B#HmpfKE3~h1LdW02flK$BEwzs%9%dRu=s(6gkdO5q4!ObV4pgf+GPD3&0PlVE3gQq<#Vu0jQ6R39Km3Uw2KF|7uld^dX7==_~pZ6&<xxse4ysv1D2aonuV3cl3%J|J61eJ&FB`YH_8=bI_R$WX_bsq_=*AN(cB!t2lo-ZAda-%+36!_;NtL_q>k-+7ZXNvf`F`s4$jbA@e47~d#qR$zRkkAVRzl)YaSDkLA*)&VboVmDGw)P4T6X~vwnhZ7%#mCm@=7}$unu@$Fh;!PVyU_M~%0}Ue%C(Imh5c8*sS|faIL(Tl3xs%_hl%p5ahjP_O;@g7-i-iJ+kSt{_U;Y77LbPS2|n0Dlj5qPkigg0)Zth-dHugHH+<52LR>`$<l`iQKgem1we2{gq;S--S26D~)`PrvX8qNP=g4N9e~!8$N_v_0bUZ3;Sn#{qWb+oSX6MS**jKD%<e_?q7QM;8k+AS$@gzkr3s6b(GOKs$HK?@6_L?G5VgK_E!(w*Hc^V&W^t9Ms)!lKw!ImK&<@?`JvwZD3#yF4Xx`EWLIe)w^{PZ>CsfvxSv%g1uQ7%8*2_R2F9rHpXpaOM?GAur<7Z8Qwhfe>K7fTQQNdmMXcg0;LmN5XAHA$_pJ&y!8|Wv+=pPp>g_F`n`1hW(&rZ|_AlnRBP)dF5@0t_6Bq2041=ypi9t2!A^QVNg~T5g?e5m$W8x3tU7*z%*eLnN{RW(7d(WxsNX(t8cD$<B+}>Rw8&;E%ry1F>>%<xb=xDeG=vWrF%m+@jb+v61iBi~pRpO8LSz%h6tRJ1p{{48v$NipEd@5PTi|7n=R_*Z(-Ga6S9&g<JvVuf@YltkK>gT((=zQ6aHBVoq<+fHnHl&?BG>;H4K59Iw(IV2Xow+ikjE?$mW8zyZlv=Rl)aN-5sdUX{9}9&GGX8sPQ8#=v^S5wL2q>v;wctP*yYc=JN@L#`2)e{R7|Xg!@5s|q_xurS)%f@Ecb)cyZ~FPkM3D5YJcuRcau<XlNyaT`83|8`%88z$<q&|7cDVivH1Zaqs1TFY?C3eZ3xV&s*mq9qQ*)T18B>P-Wy}|z9AfewECN9%o`BqguGByZWZJ|fmxM^sjRyPBtzi4+?&+CD1Y>bN#XDPk&8Pf+TP7x#KJGGrZV`Up37uNOm~ndYdZQ|yp!rnskOSs|6YCXeZ}jvmGi(H8^dDSdOi#N*#Ir(&|Ael53y0V7lM#L2|BdO-;OAfNb{qDc@;m@Hq2Tg94(}_R`Z~K9{VZ{%O7Jve>1y~{mab~AU>lpNTCmY?;MLSFyjfX^>JR{wDAY{6a(52IJIm<X=Q#Z|_>590<z{?hE8os9%&$9cO%_Nyzwr|W$e!tk0~oc?GHbcqr=_(VxPl?BhWFs;=??IrSFmEBfj`weZR9UicZxQCH0?&1(+g(KPgR3tGDQjFR8tZ&f06?$J-HN?n!<#>VIGNDl7bee+&HCcgRY=O`X?my(rl-*C4h%r3wHQM`N{%uq6zRrM8*#fGo6%m84Aetn~$MP*s=Fjuvq%ONk0$}hZ5T)Ijd_B*jo6*?1;T+@s0^Axnuc63o2kCeoNCOaYf0GRVSLOG#l4mk>83fJI?~PMmM@SWa$!efX~u}e{MB=B%iiOtYYjRF>l!fetc*$6yr8_M@o+<eB(AqoO|F*6wvC8zZ+lmQ0X;uGNOYG-lkOk{pN&blkOK1vm0ni{tE>uy890p7k}sboj)crUm#NNY!QMet<lXi7T48n9u;7qi87MNy^8liWb<@X3Q)@-MTayiCn(0<O+6buJm6BSw?MN6B6ch7Nl78a3)Pp0s7jAUtz<8e^ExbO4^EHZ`1<p)PGw;v{^jbP({vS8Lc{Qc`;OhcX(&tZBA#CYCHeVBoT3s3p^E6$bSt)e#-~oP_z_hy^K2eR$R}6ArmwX7#i8w;baD<I<r*1}@fx1E<<F6^6DR{bZZbTYP=Ccytc2Ph9PGZrZVgMJDT0uR3Jy#Hewl6s`ZdU}BED*MhlaXLZ=!k*wGPs^or<QJk$DzrzOILqWd1ggTD$c_0Ud2N1<%|C{Ml{!0V<iKT0?QR)iY2RzP-~>T_(;Ak9Q}DL-<(vD4HkID{XDnPQmT^i7XHhgrMJN065wiJQfI*Ccs*tvPHqeAv3N7%AuxwpR;PhyW>qlmUe-nLMW+<9L()a&#gFz6US7gI7_w{4K>=|*@wH!$jsLAg9(tO7V(ec3c<TVqThyM%14^WgN($W5;ffh&SnsuqOBwWO>|y1F4FcOErfvCeE`uP`giHXgL0qH55Y><3mCp4NcajL@&<7x-Q|oCE#E)tcN)etl2_In)zcQ6uJW6#5d}&_2O1H{Gua6hm!hMrNOJEL2!IvBMbBJ}5%{y_E3Uo*)e!=aAwEYdbss@qakOssL|_?PctKcccTDuS!${=oJGI&#3+l+^?~K5hAB0#MrWLXsT}c=MW5=f5Q9(V<w?*sq2|TN@LE=?hCY~vS+#~2`O$qgr?hZC0M6oCUg72BsH7f#?Ik9S<S+5N1T2H%DLZ!<wCG~8|zDt5c;(+42VDrX_!idXro-OH1p)Nwf%C54`zdGH+gGH`6aoCAQDBWao__dUX@syEFIIk`q!3@|z(p+vw%pbtU#_1z*EMBt?JauX`wEXW;s5}z5PYNvEml0K#`DLx1&)#cj<IZSNdCV0#nfVwk&6l>KUL@4ttb_Q8Zlp>wuTInveQW|YlEq;H?xf%!r($lL{+@Z(^%`lciO)AafJ4p42H2vA%kS1;u%`PnO^vdA-zoNYZx%WO$Mu8^WD+-J0~T;_011tzaL>8^E`s~)WR8WW>eRv{N#$4Vkf#)2;Z|G?DY{QK$|NM@BtN#^HvgdQ?ly`bD)O~KRNCh+V_Z_jPNj$?i?u#VpuCK|$dT_qRa4O{(|y%!A*Mou8ga&@Usr{K+W3gEZ<vzZ&(hmXfv`<J8aJj}<UVSTzt-ES(LD}7Qzr}~fYO0o2Z%isi5{Wv%ht;R`F**h&@2XDw6VJoGY8im#RB~(&$2-|9<w0gDO=#iZ*2O4O1u6iL_ci8|Mp9JFd;k%R+JOY)4z>#qS>khj2=P}`oac8|Bl=DZqW`Kpyzr}k2BlT6nfCyw`tk2d-k*jtgA6JoFJzAvYO-L<hkpMfboq|>4tzOoYegD$wZR%&S|0Ky#-G(QU=NPez)qc1aF@7Wq+_3zq<=^N6i#M#3!n+%gY=z3fF-0_!ADSd!HFTNudLILh?v0hU})1Dd-)*_C-eb;Vk{9oMI~3j4$j#4*g^qsw{bvpM_-2pNl#Xy!_5cP+ipiB?xTHyRKbo>=?JI!x!@a1CRrj3TtuegY!=g&6`ZnY^a7=<RKhKLjd0(!#C(Q?DHPJIRDr&%$LUpHachYgLlxu_S2n%zN%A(ocG=~AiljU1dzvtL`7?tAVmE>&x`X!^Jl17hD-qfNZ6HeGAR8(q~vZ=69D#e7S!R)Q5t7oR%#z{9%WkL1=e#r-rlxTUP%LN#YE|^5EaI$%G6W=QtCsZzpu{;e26*+dqj4vPrKYBwEXvM;MzicCt`KZEG6z~Qd(@&`EpZk4_Js46~;(ZP9e)T;tPsq(r57)7K)a1v2_~BTsPnE06ekHQ|FGoq;8tLd>JrHiGc+dpKh}Z`t!|JE{(_50$gC<k47cwn^_~#woc!#hDPuHb_-6O?vNQMA>1sIWBZuL6Fb}lYKM#jEb^9R$?q@qO^pDQ;~T~GR6-83C(}U5<3@^ezZsS9DxCP|RKighS=fl`NLQ}Ud;l99hSw0ajZO0<r*(j(QS^9<3Di-%=yWF#&CRJ-k(-@aW>Cn%``_f+;Gx#vHjh`obSY27lOXY-Lp%^TLXU?^ktAS_TYoNO+eb^&cGv9xZm1ce?;y;y^)b9vM;ZM0>@5Bc%vbxeIzYl0Pb#B1+hE@_!6R0lqVgCt$#<Nk=<FolD(8ZIXIow{lY|iZ;t-lOz9*kPWlgX;IN&K?Yu!uCew4$sPI^ZiufJ~wCUmsgw1EqFmsDufKHsU4`BJKY`^}-hX$&<%w>+b4{O>d4M;_(pWWW?J1!^d<>DHKJR&*iD=03;i-U(wlEn!-nsQTULc*RUxDI?H8Okh9j)z03LZ~W+;eANs9L>oE)Trck_KWO*J5kSs?j1uzLU0_Fl5r-XAPRHljxJ<R9=7V;AdC04mMKb>JnEM*wP3MMVXT}QCrawooP50hK&vBE{JZ<`?_NMWW4*Z(v$8HCioOignvEOIn&hxputm}-)!L0Am75iX-L+E;TL!WkV{XM>JI|g0rvta|CXnJ@m98+9Qrql4=g?@<3{0kn{J9*Ye-p_|Et@h1gW{PF$18h1ZZ5jn`qL`(3vw)wR(swh$2ep^NhOXh}@H6C9R!zk4E*<7kB0pS3CGHELL1`UyZn3{_Y`*-!e-Xq+eh8Cm)t49D%pa!v7I01B|8C#w3a_F!8E<jgct_#!MD=GBnEVSkqD8Sh>Q!FW%RUq4$B5?VX0;2ZyEF1Ru|LQse$)DXAaL}Zsa2v*XgN><ZNOjqvF#rO*Z8iV0<?#_yGV1+#UtIJvv2G&$&*Z2<NNSR#Gr3%W~krR646+$bJlZ9H_t~8WI;bJtDVsFp8nOs|Ehb}?$mK*QS^I$h4<>QhoR#T$U9`wV-Z3U@*)B9nBj1}4H$5E#)bfyCu5u+aev|cC8w$+*^+IN%wFC5oZDmc47R0`R4SE9_0l1T7KP)?V)?qa|Koc1a{J-nq2t|OkEi?lcgsQbd~#b%@1-v?qf()EnlIko%&V2FWf^3EmuqKqni(DMXG-&)H(b9N6sj-%yB`lb8Thgb63LJ8S$lR@%};8x@y?IkAMe-KhYwcw2?TBj&X<MP;$ru4b2^!pn!R%Aet+CtoVLcJlitOR1sm-A?(D2`Te9Ds>i%tM|Dfs>PNw;ngHCaM|LlCfP}tfVS8kq`gTmg{?oH*#-Rs`yu=2Pye%&5i4SU1Mc5nCOc-h*?@4UY6b}sK9?{}Ja<%@1<$KESW`Ujo;{>8A`?Rcl%!&3k7edkBTy%`;*+xI*Dr`G#k+N;c_@72*o@yAOu?X|DpA3NQf{QmN}Sb6R?=asEWy1C^}#{0vEcHzf-(b~VevPQLeKfibA^wziU@3uj<Z67>epPi1bkB6sEJ9DqU^LTdHJ8d02w8oj&?Gl_DFD{xFC)3XM)?Mb#>Cd~Zes5Gdo_C+OnwiU|le_)9%4~cB8~fg{a^2ir?=RhxmRmf1E=<lc$0cXIf6(54>AucR%Qv;BdwWpb0?D(~t1L>3F&xaRkM_a5IVyLo{=9SZ`ttC4TD$R{`nCM~=(MnN`(yw3@crTb<$2lK>sht?_w)VQ&FkU&&B@8#@xw{=I=^4aSbMF#)BNK7wX+!QEp84MhbO(83upgwyq#}8c3&qM`*>&QKHc>5^Tm(h?sI8=->WP$<^D<K);`<0PM<v=zLwHnZGB$c1JSv2RvcLerOc1=d(j>|EUn>e-oJeBP1lQ)(qn7iKD_Q<^vmnJrE!0r_NF(F2kwP8eIKRa-23DBbSpjGx_li!4ev6K?$hGr>hU6TQ@MTY*{xRhqHhmxq2SrY?A<zeTi!gs9X!0;J;G_^(yH#>rc2Wu>v?)rYYyMGpD)kff%UZet<K5(V%)pio6j;=ukF`PVZUv+uB_K>Z|e2kgG~CiFdR-FAKd(K+}gh{53AdEa4@dEc!lG$!4GHDK5z!Dfi->g_A>qQbK$YHQ+2o3Yu7jT?T6Y)#@^~&l@6~f%~AXP@j25vUS8)b_V#P1e|K}%tQ>fjmA}YLnx{{Lj<tC@-`s!fb%AEP*2C@X)9K54>F)jc>2%b5x}NWEzML+0%Cm=?>EX-C>CNQi!W+J~Pc}<ecDd>ud)K?q-R<(0oA1GiYqz$1T-+^t&1;as^LNGT;dF8Jyli$$yVd11Gc1AV;&w-8#pS4Wf0nt}Z1wY-#X;45JKlP@=-xe4hxvosS70OQo6?!%yj|2b%Wv({$^G`--Z-;cSikMqSJyX>k29xGI@zgid->zZb7$7t>Xfg~tLJAwF7L0Oto&6w{e1ZF<7PaoRL@^F@AoFdV*l>FTu$G-o(*7Bk9$|?V&Uw(Hmz0c%k=GKZ?C^~;=Z(R_b;A1hp+aDwex1Z9S)0ky>|Mp^VB}XY|fp`&9gV1>~HVSt?hODskxr^(ydPIA$?akdF%`Z<pUS*7;kS+ABIb-yq@0KS&s7Sv#WV~`}(4=-g#;Lc)HtrTkmy-z1`clW5?c{?f30@zqhlydD1_1HXjBroA1}7`@7w4?WHqlZ<h{FYSZ=Rv$wf_vE6wXZ(VJIti3&LoiB$6r(S1g(XT$gpKZSN4sLc@yUoYzm&xtJeEQa1_pJB)-r3G#eEc+dY7h2`SIyzx<#MrpbK6^Y%6%_0Yvv2*<1=q}Fq$8pcRQU;XZ&`&zx!jkY?soD%kyIGuzb1x<GNVLPZxvPX>-@vJ}qvopSMbxo0qG~_JLCvcQz}ITe&_@_qGpDOZWEU%hd>7mJdfQ5E*VBdZkM9<b3PCRyrTfyv*I;>f)xqdtIL8+xh+G{#p0NJ+{lvcHe5Ap1+U!=Ld_^z1KqbZnQqzdw$=1=+yQvYY+R&$HMaXefMm(c)7v50N>{GdF^CyUa_aIyG3jI{ItEE?-y&gTm9+!vRgRVd)}S(I<F42c3y1F4~mx8@(%ANi_O|zX0to2ZSOhm@&3!~@^boe(A#;q98A)qe(BZTpN=Y3$JrdVCauHL@}~9r_EPF}=IxiEmtNTEm%{7W{BiGM>1~$1=i~G9?ZelX(PQoObn@Kw`m@f<{JPe<-zm8F_I<6e>s0RDcJ<-pc=V(BqX38Y<G1bdW#O#yb}|RH-LtB8d45(n$uC{+=C#;|LtArm(6e`(gVAXAdVPLdeZ1>D?-v)>k58BDrO~WZI({FXFBW@?S+{rmqckg=pYC34uHRmtEcVAk@9|=*=`MQCPWtL#dN#KY?yui^n_IoJ%KPr4_0X(6x3BW0bn|L?yEX2l7uW5_gWi^Hr5`WmJL_=zegige`Sw(K+b%ZyTbEa-g@?nZe4+pN<MIyJ<;!CGYG<cuU$>`=jMoBj(k<toO2xaGm$nWz?=PM)FWRQ<HP`Q-C!@U|&V$w89Sxr@nx%Yev>Z*{>v3lIHamE(R=e}<!TZ}=`RrgmUq8Eff6e4OyM^KN)AQ`{AwAi@KDa+PXgBZP`mc+-A1AZT{%E{;^_XvZ>t|>8Ps`)V$>za)z1rCa0xCT{b*?kJ7xV6AragW=>~^fDtxEB-QgKGb{NQ}@-q{>wZquXIj%Qz5&`P=d_+q_W+<2FRmp$vY{4jdGzA4-~2baaC$I)J4oW3||t$X|Dt*6=Uw0C^7bvSQM9^d!!wL!W+t@PWK!v6GZcXr|}o45B@?auYn-mLp{=}k^+Kh9hIi$(KlQQI$VZ4MTT_H%#QUtHBH*5#o)T<_mpzYhofi_NArS~ja7MvaU6t=+ThW&Y+7PK(8hQ>)fL+<f>k-m5upuY38IR)6-kIl0{Xae6k}f84w}*?GuccOD1Z?>D2T-Mhp7>vP#Ic5V(|+FRH2<Ne{?)9k(WhQYQ)`^Bl8Znt_*o8_x%=4O_@8I0cC-Rr^+cd(nkd>*|l4-U`nC!^k0dwM)e=evX3t>bRzb>10V?iQ<=tIDL>TU_r<o~qmF$H&UR?w_CCZoQnoP0tF0i%iwM-A?bg{q@b8v+IY8+4yzO>UFlB=abhnH}lfGy1cvHK7Oms4<@&*?$$wX*uB`ieSW%|SMuZS<Er(%`?hm<dT={$rVr0vD>sk#r^B<?n^9)#cK0-MUp%ZhH*V+f_~|15!*0)?&ic<kZfgVY_Tafzcsg@3ciruay!B%+=?~s^F1!<ad-7v`d|%1;_c!PB$KLc}Se~>W$C<lnvzl4FUOJQc;ok1e>oaVu{qD_~HQe3m4ELJ1?nUuw=koocdDgvn+bv%&+s(|@+<5@Gd~$F#arZM<`Ra5!-k$F+j@u^#ubRF-ZQrk-lv<A$<;?Q!;C$*GycF}foW8cU*0wvEIT%}Q(;hbMcH3?zM%K{IYTnc}w28IEZ}gPRu4(Y!aNO2%+H^Uwz0~!%Jsa3H_)V3PqCm<tJI=s1O(@yX|0y7E{KQfpf9O=sTY9M>0r2Yhz5#{dnZCK!9G>0>PBR6v$apw$ZO=<IEzjQDNj3L&@MYUh@b0DT5j`f8tOxeOZY8i5^dSMD45d$oregs;DPFgo*3>jvRo9-*+!6I-ZLN^MIWhClkr0CPx1C-8$!uApw$rwzc3wBMyvXg5T>DKch%Z`RYq>W3@H9(nFJZJ!OMBQp)Y$m7xijsJXH#gs>v&V!bw*vynWnUg^gOiswg*71=@@HIoK_!dwm|UfPsYw@>S>d)JGGkd(rUS551!p|3(y8oy{Y9+z0{i1(R!9=O{Z>xjSX-C-O%~eN;8u3>?x3jH5*J5dN4xs4N4mttDJ(W$u%5CPI|+zR`@p73J5R28qbhYD?Zh}Ljh_+6^Rg1yg+hoEnm5=0;~j$QLp45H!9FYXqJ8#%9V<4uy42dQpNvB5AxUQ`TgnXRq<p)`hqZSPRiFC)%)Y>ZN5-cY9Rb_<)n0zul(>q1~1l<Yn9?{rC2TA)bh1b`NljgS4`+G^!8_{w^{nFbXvR$_dUxV)G_!kSB@ncybbyvGq$XFA3MsRpXy+P_CMCDh4bRc{Z;X+;Y#zMz6=VB9*NmdhV#YbN!3;jbme5ix78D?e1!2L6q8fobT$E|WNDqT3%tM?Y4jczW*8}T3m9UKcL4~67TBMJL0C(!iBA1#IR2=TYx%1y6NtB5sR?NGXI)F^)vI!C<NU{QrF61UDW2*^lJ7HK&oY}!;DU)}R-+$DZQ~EzTe33q1s2uAbq~e2cJ~s^WU6N`+D;ePSV9K%;*Cd%aq}e>NYEA*OSNaiiHC&^%>y=R_U)yYtGP29p6!X{0#|jtTtYW+HnX@laPF7GoiE@kqWSU6ZP_L?lK7(aYWcoWD5ik>xA_|%2;<~z7!zyS%WAjquc3({-Hb;zgNk1P!}ys>CjGZ5=(|0driT5t<0jaH7?{0)&BGk`>5)~L4krMID&n4Fdb3W)SrE(=`wcUwr=Xb5DqtzBl^Rd%Q9_^VNzL-KPLK5q|MStyu-kzFIHNXnBbSNdT1Ki5q0>us1~adhNFHfpFV*puqgDc6!gf6xC-|lgUtD`Kuv&Hk`eSGe4+<u`@6O~;q4Vg!50}_ShD;iTKA(OIrQaaZFR?`1aU72L_8;osgs`W(F&*SoPW?ovFk7LiVB!h1_W!4;XaZh^s4=kL>_KiPGITbu(CmPuH66FcgB<^QUOGE3R%_;MrCci)%2(f<n>eKLxq(vvNOEQYdS^}VN30}ZhPCIvNCq86s2x~|CRI&%_Ml^MdDGBHS`W(Nf57&+GzuF*LO0E!<%~=dhz;bnA^$UOBQcEH1ON+m2jiwS0H&q_lZeDG!G!#^m?GY60%J+0r0yym{fs)BE2%SrEFe-wt+BNF4MNRHWcme_sziDO^PT&QJl8+txiLcwQy~!Hmo#`J@!4;)Sx%!N68*dAlh69UL7nW!7W@0k{=BcG*}sBBLlhd22%kQQLcdLzaGbN<H*1i9|2>Y0H{fvf4iuXv7D*D>7C0Czf!KiuPaBP=M6xv8A(EHlL64oOoe(vXY<MGp2GgE+V_gRTSobZmh<@rL5NFe{(uTH$KM_1U4aW1}q33ja{ON<K0H7>10LRWLECZ(n$4Yo3H>Ukl+gj1sC}<4-eAyU)^QJSDILxVxgaG_2sv&6ne!<ja(n+Ka_jiHCXd7BaXbuKmsNT;^HnWx5PwyF9*`3sOdfV8}Zl^K_TgFZn-tX@jyYP2+KV$5{-<_R(LuG{f@OFD|*Eqnx0Ob(=ZtZOunKb^`IxsRB{E^NWnJxH{**^>c$ZX^5y|j_R1~WUmMh08VY-c`FU3;3K?i*Srb-2gI(6a^|==iJlyF39N9OvI)2DD8^7~ZQ3brq$@=jf+7-5}_}EYQn!dilX=cVam%oc3Kfe!FeY04nQD4b6q}@CQUegBVHfwo<yrqX?b)8OobW=r(Y$f!&%q<B^x*(hcfo68I16Et0;Aq|ciO=F6d#lW^-gHla6?cxb^l_!wd+&1IdJ^=uo<HtO)(a)){Y6;Un~ZabZhhwHB4w~VE+ag=Ofn9ZdG*D|ywlm^gPszK9(!v{f4c}K`d%H6gitG5|>V^LkeU@}@b_LDkb8cWKW08?snZeZ1+3Seo2uZ|{rR*1r9MO0rfsMgHI&CYZ^Pn>FSyM)55(cgf`TIDJLL8X+Zxietc3n-1KqY6jbl{1<x{1TVWaZ0Q894tE;*{(LVTo{zc02v%=b2bGUvM1WKXQN&Vs6+ECz{0#|!<@h#vH`GRVrV79OV>`Z(P)ekpbj0~5Ml1jv$}Ths)9cCMYCGGI!&hh9{W5CwPUa`PJs%BDW*|Fj-SxmmOFPwdNRV#Tp5urtjA1xYX>m+o3`;+{s-_XHVUmUBNnb4q6s(>*w*kbZ;Ks&XKn{11hX{;>R1rv@dLe~&0VB1H;d>dqM-Nugu!Qlmtm;w(~_Cmmg`zeB4W0O!0XH65A&&9F|YI0%W8ejR$;%MZfO5cP8-7CBY4X2rwset!sA4xcv^Y_#&llHpA;)t6&@I&_LK0lTD;9y^0jishYAyn`(Z1o{YOsA$a|CJc(#`JJljRw7F~DjCUib>tka$LbY;o3_#$zwxe4T*XvxZs(Iho$(@<Fx7IYWhb=>i=38ElK1FV~^WPJm=(SS1^>e_QqW(KG`NXosnR`mA48q93o(>81Xt^*4P7KN*liEF>kfLVKRq_SGm!4e>8Vqnw@J}&IzZLxBF&$N_vi=pftULTigW}$pigm)>|f+JHR>B|?4sX)gE&d3^cQzKY7iFcK~yu*GpViLh0TE>PU)5_J}1)~|j4x?bv(C)Jm7PbFLye<~E*g=01e8k;H*eBGpM$5##w1TnfJ21xkR89Mrmbiz1-v7Fh{i|(FEs`SFnBZJOsl=Hst2Sq#s58lG1DLxyjhbPc^F(?4jONFq&Hy$@_~O}9+}Ii%e`qx@Vdpm;)`DZBsBO1td4lZ2|9Zo5-_D(aEXgn#>CbQ4wKZ+^B$Gx#_q47JoEo(fghtT|*lN6%GXdOi=IgXQVInA`s7ImRz={MB24syw`R4ShRH&J!m3#qdO+l&Ds|2DZ0f1A8DcE3f4n4*~@GZxbw0dBidpk+gq_BEE&U}3wqKVbR0$I5^LvuK$BJ9oI^Ol^sNzK=thEH}5ZhcC_gfy8QqPVgh*+mjdL=S3K33^idPi<=#1XlXHC*x-+PdWhf(0By9G}{W~d1ppF`e`p(0}zUtmM#GZg4JSMaE`4MZ;Sa_)rcJIrT=8Q;R;ZfF*FwjHy#QR&U8mzG1CW`bUMAC<b*pLO`*aLZLkQ8EfQa7(>6D?9X`ehQqfjSN#0|A3Ii^L1z9a?FzH!Mdm81kdOEYUy|cTwe{h&TE}Rrk&(2F1msi&}<=eYTwRZpT`1Iphk7vI#&qgCCl9G(Rq`!0nnQ`8O<25mBDgoynHs;@1M}yfG+n|?l-BhbLcHjgqdJk0VyxY0Gq#`u-iHiYMVRuwv58tG$b{lrPBkGV4{Q3;S0H6-AmqA-$^jOUnOkTH#;nc&?c=GDP;puI@h@xaAg;1ycp(kO1Vnrll&PrVyPqiRtB{Bxb3(mJ#^luKsGp%(XC=k}L3dh1|?0C!QR|81ocTFk>P{*@YFAPbP;tlh{tsZ{FqCeS+Prwl)p<8-VEI@`jT-mTaNFQ;95;3~SYV0m7zr@ydlC9I7WEe+ymT=f9hK9+`vhH{eJWDp}A|V+_Olv}Eg4604d@q0_l{0GNDxQo%DuV+ul9kV<WSNvq>M@cVjwPtz%Sm)RVvQsl^jiNPBVDFHoLPX%WRM+|0OO@5c4AY7CNXS5p+9x#6KoJVL%#zYNoU>QZ+M^JH4YtK5(eUFP#1XNb;TprMnu7DhA*9upO+KZSOpe_4JF~<kK~$vG@R3P5^>SmmXO~X_u!O+$J?Hp*xLnc&V$j@SpueTgk(1dvLu2uw8R|ADk)y9$)sa1g2pliV~$u@orQqN+!PFFIAekzu62vm4cbn|sBnqY_#~*b4;ZsrAU%2x0@RTJ4L(ZfIP?y6DoFXsuo{o`oS7aMfgTG+v1RNS2T7*P`e%;D6w@bZaRem82+Jfq!Nd?{0!?8zQ9lQ-I{Hu6y$=&QZvM0vANr?TTQJ$kw9uxj4p`8S7_>}k06ZA?pYE{Lfupx6y8T6O{a#YH!rl8&HjX(88krTOLTojP_PIMAb%8#=%?=sgizg|AfrDOx2yTu$9TQu{Zf622N=N&=7436qA#8`lgTojKTD2-TCyuERUJHDesI%t=+99xRbS0!Cd_PzMKE=nk_ovB5bmdoQT%jOoEZ-t>wzD*KJ@o?JMU@(|$A+Y+!!9bCM!!=``<5lO8mSsB!T3r1LnaS96f$}Eg9V7(aOm_S7@f{%57{#Ng*ZW4Mx7@9g=b$BfOr1yFbq^Z;*i-K<J>~`5iIS^0qd9QNpnRt{U_dk6vj;-mBie2F0@jAC(1zJ#L8SB5HG$X0q(S{)W_eVLh?^`8wETG*@N+9Xk$-7Dtxy`A>D!Z_b8U=s0xh(>g3}}^f>ls4jwzZ+i>(q%BP7;h+Ui%A54DkVfFy(imCxN1BMgYqMCzo3&r?4ak!+!Xs9%<R{zAB3`7QO^s$n^y)9O%f$<5Llh*@b9JWb<3xYiBO$y{T<Ob}mysYYpuy`k$V<H|o49I6(z$|-Bf1>+=!8*v=MU?{6R@_>S`(V3H$FaHF%Rm0{51D?NK;vz|IZ({Tcs6Z~hqg8yPc|;KH(u=s)+E5S1~XD04DG3fI*}9r@&#SfMAMaNa@%#wn&c2wwMxEFGz<A#l!0eCbiFcb`FjIf=<~<&?WHs?SU&~T`2ih%0W#?&0Pxj0@Y{b4ka%JInG8YV)12m|LB@5gf%IbP!W<yg{QL;S_Q81EpH0MAWZBR^_z9h#yzjNh*bqR?##twS>Mi6f#GkDe?hMRTBJkWqK2uM$&`xB?;OebLU0<U4<3ny050<~)C@sfGs}kfGg&bkP6Gt<t&EWIU;>ULoR$Ki<UGDw@y8q7i8roG`UPGELXXw!ZbBg-A;HZtWH-xRqpKOCEe?9qty<m;PVwr%9Ok!VR10fm+0Oi|yw%>246`Vz#)j|6ZD_(aNjbu;^{d|3XB39Z9DTfv+o}qGJxI{oNjfgajW&X(d#2ghRuv0CMB$bzBBk~z~BczMO0@Vydl{^-lJd$gFZuSAbv<AjaSSxQfUj!4FP2}av$Wo;RwhruBStP7{9U5*Nk(m+Rp$<1pxS|~ESR*PkGJ6jnE$}EZ==sot?l-b=(i@@@EBV&N<q8RocQGNFYZ<(_5lm_lSWeTmt^W7d{8Y<@R`8KFrCHKMf9M6U>M6Yoa7yoc4oDph6}_t6TRPv%3(_YvV4%?vzd5Zb5p1xqs2Ul4a|3JGY+KqQt0ltZ#rhc~JEyJCUEnt|P>Qm4RmxV;gN#^Ta<LMz&zj6ih#`wM31gYB1ZLf7i>i$AL)~{K5purv5$y}#A{3nSg@jb%@2uq)t2%3XK#P1jV)j8MDuYPDH@iyy<`PaFQ;R(oYx(ec!m`5t2|vXy2UUFj0>7mt{vA;&y)7MX@=TS8b`WcjAiZDW#0XkPGJiERlfmUylUFy94{fdY=(teN8d}z`+DO%Se`ga~mblRdEQP(~sFBZZZO*>AQJ~Ei9du-+1*fg*SJ6p@3^!RmS8Fm;ZKHkXNE;8@Vu{_STI0zQy;x-hu)^ZG(#1!Mx{Q{7#{epDd)TxV1WmALT-kU89I>&z+wRy)oJ0k;e$dtnQw#BGK!6|&_8Z#o*ztNlX4%|h!|S<#M1F^U{jMcYr9(hMQzm9bp>5om<d6TUW%iI=ruoW&&9OIT)B?1UNJ?9T`UOi%Sc;<|j{BM*>v$RpTIjg89fcg*ts~$4OP*ifyo;;D^!R$ErSU6g5%`axa>e4ob^XP<;k{#{#qu;z=B;O2?#Pr5a`Jg9dU`^N+TMye4V6xccs)$1Sd~n9tE`G-!y^D(n@%!!X6qmZurJ0l?>2reAYb6%bb+#Z4(0VVX+xEmBttq2USfx(c1!_U0t~`<Jb?4G`I)B?3ZhbJYcL2Fv)lc~#BPa0W^1|_GA8@;i-Q?#i_s&PqN@7#36kwT)~xMJ)pIL|A4|hO(}hpkbjTKyCpnsuR0M#;1f}>nMiSpUc-oVT2r(}Z@MOk7#Y2*=kdS6BlwKM_^u>~EVUdb6m4v_H&Ro<zibn=3Bsee|7jb1t=j>Ahxlduc^#<2J|3{8@su7loIH=VcQeJ?$mvsYU5S=jCw`4LTirWK#EXZv&Xi*{-w)hwWJS^PFFxfEG>EKK03rrJ)5TbpQQWuxYWJBy-d|r<HL4-aiV7Mn8UqCcHyqKZ9#2%>?k?sS1sb>v+X9JV~o6vT`;tqD?REN$)urW2^Z}KzIPv;>&B>j=+h;ZAZ2j3QLeF;eT&MA<ZQApqDk~2h&uYZ<6eLB;&2w``b<T~1WlQU0RL3|5@$+$OhvXKnLoXg*k&C!uPW~bv|nex{cMiNXKJxNMdV(Q=?H-UC=(}7x3o1fmqsf}?g8hs|^yCgz&aviQO>Y>S$&iqt0<hK}AF*|$@L^R!-7^IkG{ikONPRw#VN!r+Z(+`Twa%}PX6aty$SdwZW0Fw(oTt`nG#CF-}sRW_`XV1rnq5>ZEsTA*au=fwGsp~AT$$o<MHJJ=o6jS*{Wh@r0%;x%ws#r*=ih>gxmW`HPpDJ<stOeN{n;!2lNVV)oss_^xpX{ZbV8V_9Q<5*1Cbt5J%`a4pdjgAIgSd)Vmv=?9q9#7k_C{JpCWl%%vpH)Aw&y<u8XZ4(WcWl50FnL%I(_bgBm_2p@O*!cQF#Z!h7{5DQ-4FJpE3GwKS0k)<(t4tu1lcNvJ+VqD@@(gtYDo?JTMTL=X9|h9)Mjv&u{JSt%AV`e$<Bwj6twhz{cT?IJSznQ)?#PqF#3eOjO(X#mcvr2Aek!7O7XVo_GZWMqo%aUr4?me%Pn=W-Br^uv@e3xFT!_K9Nrt%8~s1-gaCO-^L503%)WLM^x&&v2G-1vco(8@!i+qSUW-WDmA#Nb@b{_{PK~4*U$o&LnB|2ZgeSz4oWLAN;=jL$0n&UbCKXA)YJl^-c1opXJde%p%K%~QKuRIZQwhcK=G28aN~0gOu&#uEuTR*SR2TggJy_LH{v7kU~H<-R*+wSNBk~=L;}HnL$IUlslmjqA$bq=v)UUwiyQERRZ2Nidx);`e9P-AZshkSk5lTS^ce%6$JVrC+BTubr0c^6xmbjmP^nM2OyH>ooluuS(gzH96DCShF#1MHBo;N)nS?Sa&>0EQ03d3*jkX*H`N;zF&#=i~SziC<%Ag7LV_iKO0vjbA<&wUO=devesf->%<0tqDdu~@|Fd0McyWNq;p`j51&X7}zkJpBdHzbc-KPZLuTVAKQePdV|k~uEY_n@_VPNQV3K>gTzz288;g~^ifnm~_~;H%XT3)GO;oDnQCuK0|uCBA~>u(eO9xS>)8tZ9RhHCMaL6t#vqAG>|qO-wCMsM8oBb8_JOCT2T;5^LP%VaCvnPYU9Dh>liJ2hKYZ1hHx*5O(T@{Ah46Ce!AuB6LRL$#{F4>MMc4T{Dk4Y@olVZbW2H#Io{Avc=hpmOYtjMfw9WLpF5K_81?TrV({Ok}3QLj|FXV+KksP2?fFtTQsZoOSV(4eQR(dd;YDFsk;V%YmG;=xl5e~d2^cnd)y~(eRu0Xmxaunj8GaaRam1Q{)_o_{(AQbW<aP|^PndY8~ri2%foWsWS&p(;CGk2n#ZHTQk#x>z9NdV8d{ixNSnK`efcB!%2FS)(z8bp=l&hD{A$DdT8!l<q~a^PZQHfmC#d2R08O$y<0(|UV76w(sz@Iuw$)b+4>066YvB0Shh+AfFu4ye<umfV0sJU2u3J$Z0+K7-2<9N#kLR(z1pHIfp&&lW>06dvrUD5<ahd*sKlpZ26C`sbK=)5tHGj#`!S7nj?^h-jFGxmu%1^qM+r_i$w63FDijBs{ztAV*-@x6wK#ic9Rx)B}AtK`{q-@lvV}9Xc5fuC-XA1#4>T(eP`<M29q`mMp9-tY^p>^X^CkO7}@Xr<g4Sy2pWcL38=fN0S*rl~!Z_-oRTl+;Sl+o@r6p|!R|Dw?D7DygFo-&Vk659<;DxC<qR^YCk$iNm@y(&L~%vLL+W+EW>BkheTZIfjG^Gh8`Wl+m6u|TUmbf$udsjY=LjfMt@FarpiRe#Cn;J?G)V1=)NVC>QkkW|p@t3%88wcGnzH7Vm(Lhafo4hsz9aEg?D6XIMKKGikTF5V9NI;?PE2r-$DwN*Y0N6Mz?(<?q3kts~ZqU#HvAUr&nUTf^)gnr^6bQJXrqN=c_A}mLOnH1iXF)l>P1}PSNxX`CQMqlh@sO}ei{14%>;|BxA??j*Z8i|3h&Y+30<G4dThPg-CL)ar|mFG|#-=%ifJ{?uWr%2$9v&lvz7B}LCMeMQh+4#e&G~CAL>s7@N<7h308XvKgPct5|@y4|Xu(ep+`HKL<oLv+8Kk^)QmXZX@_;usiLko`1Ev`xkxikC*2fN9@X*sAjXu$-77(8<^H7NO?I~aU{c)wku$HT_B`2q#BF`NZ@w$)Ar_Fpo|1OHyvI4^@e;{cwv@zn`Fgoj4X`(ZZzJ3(64GJDbxfTvq6Jp9KF;ZHX0pDL6lnfjKWT_QR!IE))SVFD^{{*-!M0RI);dQ7+j>xG_`x<dhQs1o-NQLErF<+pV7TA`=+723_2GiaOafM<#ktOyil9g4F<*dt`Z58H>SB56(Cgb3#Vz-FhdCmC1fg|1L_0EP6<Y%t*ZD#`yN_6W-tY!BWl=^_BK(_Z*TKzrbH9ZZ4yg?k5V8)I_P`PvDDR;nrdI6i_DG-L9kU|6ZMAQ5RFz#@||ZeHX;Kbj3OQr^aJ9UdPO;$-j4<PEqK*(Q6eryk{$0raKV-j%?TA*=f;FO8l*3||N3cwST(Zzhug(^fYQSi#d7%2}-I6vfhDo~$rF9vFyou(^rx=*ftSGPQ`T9hm}k2!*=v;ZvTyI&akX9RTnuSW-c)^f&RE!qYYyefYLILk<hnblvYXFKD6;-<Z6{!+R7Y>3|`=z6n=f#bW6c(;*^X1DpnIhQQPB#@Eo4Kw4Ily`lKcz#~?5SS3;uC`DR}M<&jskDz#DOA}7A6mHb`T+eP~wn8!y4A%D`{MyS3*%IZ`NgtBW)oyy<&nVymCqpX0+U%F0D}3T%`#ceZ__OAdx@F=@4m_u<!N6o6$XAAvK2h91hndv=CxO6rlR<(ij{@_9$^oU>{z+VbVD2qb@;*Y&<QG7gp^qYOTgwZ%s{TQ*?Qq0wY#ernk_z*Hd(;ElZ7*P*>^1cMr#Jo3<v7-{XO9RotV#6eWad)0D`gQ79=CRE_6_uI98thBn&J31lA4Zbz*1v;8iflIjQ7Z&QSowUO%k*<v96I&FHN;5CM!CmIS#yp_|gT!Ob3e=o-+6(mLxoHeX-_r(K>hguqiMThr;u!`OX#G7YsZ71xVO7x+F{xraqMPKLSa_yIC|oLlEi{V*~Jo9-0M^u5y^+Jr0%<b)0<M#gfv;Zrd4IE;11r=aQELLp76R1kO=m;`2_d>OUB2^ClX(c>E&?UwrBLLr(Hutt#lFLUz3Hd%B{MJQAfMH7iHm`uz~yP(pG=+lq$_!<SM}X%8TY9nHj(G@kv%4o4^J0GMA+m7wJsA=x^qx|w*^a5AvlF`<@hMre*fGH`~oVXkyjD^^a6mEsMEmDl;FWDuGF`$@uyCO!*g@b2i$YL@-U0;WIoAfa~3PwXv=+~n`arX=VO>UA(ZSUrm>AKp8rZ}3Yclc+5J8|*u$&I0W>6U&8VCFj&aWe(<so7%&_%cxTwjOSHqfnmB*yfw>}lVSy>GL~+iC)bxU_=_a{$<N|Q_@v)w$z`QZcYXx9r5a(qi{yJk=Nif5UKo=(^M8x+|Ksiu<PC8zd?7x5A?gGZ2jTf|FTIl%*)9W}(0DYpS`@4sXs?6xHYWyCyA{eewS1vg4R1l;2$K&5{}Q<Sn^LU`G$(S~u&pi=?%c#3zr=X?2OvVdOUBjTl!E{$V8#DtXs{YQ%(#x(5!=qtlRSoeD36M#CaxRI*`MTX^L_}G<thhGADZP3P0aL;r>~z@*5M(;mU`Zy7Cz!bMx@?Ioe9Ydp-?7!V)BN$L-$Wt@?xl~J;QqoB9|M)U%3HAb2+yWX+FqeK%o7l7a9_9V%S(eGdYIMo>1Im#?OPy;nr8Gpt#mjR8-apQMNq3ya7}E5+6VRV95PLXFjkz>L>8Y(z!2e@FCVOOP3sDgQBuCmhL;r%55-*tdBZ$Fk)Ac()|Hfl14`Rg){2<Ce!Gcz8go)kf(~tuYtLTFgqDzgu{{ij_KuJ{g9y7z&@9b4{Lt+a-kTo*)KZOsGYpeaWu~QjU(+n2kmmRq@Tc$rx;9c+^1^)O5FeJJ(PfClv|a$5CHlT@%tuAT=<8awlDznobws%yChn-UFXgA@Y~cH^{@%s8F{l|LU(5^W7gXE!=34d_FT^z3d0tr7{<rOlc_UA=1BZYI^?t$xkxlN_bX;C3qE$akrC_=L<J+uUN*vzC_o#28@y;>z=8dMjiPUx)?hs9nplLlV#HNJBvKt2sR)OcG*qs9#M;1rHkpSsN-_z8Vi3*)(62dQaM;iKOM}Os8CWrRWGw2%a8<^uva4h)zXF=hy~di~*!ScgDznOl)(Gz52~_fhH&K6$s0-L;5jtP<CpqjcSo*N8iD?8na8ih2FiD6l7y-k`QF>76{5@*rB5+AHB%ak!4F9zp!JmuUM*W?0^~On?i9LyKC+bRvvs*51yHk0l<d;A}1`c2l7nL)bsqs9{!g$NUIHJ{ID0QgU$VscUZ&>}ejX$h6-FRaw^^wZJS^_}XhM_Le@SFk2@C5P6fq{f(Jea2NEFqF|E4ONQD5qU8$mSZ%DWfv@A3ba2nT?;=@BrSle{pdsB27TM1~r(Kh>-lEJ{jUtoUHyY8mi^^J|)WNj{F=p{=|y`ACXT4BB#kx3B0}ejvM&)O;Jd}N0#+=SEXzekv6;|&6ZsaGO4JoQf?YIc4qMnOy`)ragqXJvt4L@6B+~z$2hcbSrec&V#DtVr&4l7KnJ>3*pf|3p4Js7Z`miA);XOPbCN-dZ+58>m4}h|dmeuk^nJ#PNVNm$OcX{79~^q!CH4&`yVb+2KIp$^Q*MBbt+d91b|jtJ43oKgEF^cO?JSn;8$rj}*w#fi*W*r3E6@bBL?>iRCgB@(24P_ruLcU~5@7S}rKF($)MIKo7T(n1Z$8#n7oiylKCeZ>=zl?43xU3`_ZxAb)l!^5l`aZ3O6cOVNZROpx68w~i2!{w*Z|Gdz|46V7QRpU+Hp9YsHM3>q6w&I3O9@xr;Cy7`XO4tcN=t}#9-XrLZuA9TVNs<IEZDy2e#AcWaPk3eL)DM7E&xy6WQe<P+_o?|6wvw?yq2dMaAu;DIWdaZi=A@zqt`(kmMg0_0h6gt(j}JmcIWC@A7A0{ChCGaR++J+)aEx610Ef2eP`cZ&GoyiTYW;(i#&0CAA9oh=~Xon+)j)y;i)w#jF9j*Ns;SzFQ%JQhj?dNxTLGXUpQKokDRSwzk1%)DK#l48}{FQ&0zuzQT3K8;s|sG9KPkY8MUkgd?c}W{#A;$hCP}BWns`xCvjTthu2*<nBVS%duaEh9E=kvE05YN(v36;w}5RwRM7VvO$+c0all%5jMvYj|;<el^N}&&_!X86@|Sfe)fETShB7&h<cDToW15wM!tVcouz7IcQPU-bf3N^o~W7-sgV5o$xi@CpRC!mH+CshD67s^Czu_1Dkq)4=}rr)y+Q8iIL|h-A$RTZ(4M->zbE}5irb=v#44qz@Py!WrR4MCX$84z#osxYLM7Nh0?moKmk51Hu&I2g>AGcBz#R{GM$ATFd&Eqj2G=!UhlHwC;^%l1RwP4B-(Jcd4J@WD<fMSAjzCEhlT8|Po*EkID5803$YbKXcy$6tB|m;PYJ`8Ef8EO5Z{>)Fquy)-{rnezk+z)X_fmdHsSiVr8w}6MthhNyN^RgsN)V^hiDGt$*B-n4@|zU7-$ky>BRwVZC@A#qOcK0KJ)7MEk`qrEe2npdd$&GCnL4L+3n1>qPZI%a{tB?>=fDsnp~<mua+=cQNLlDG$E}=b6fLJj&n*0ZHu7wHWCq!~)ieQECb`0-s$)-VO1(g!f>a4A3@nzU!xT;CC-MO^(1OI`jN*I3;@dxk`k@%U=<rOeh$ZPXfhuU5vlGJoN~DkQZ@6#*>4L&&)bAokv43+Dn?#QwJ8jR0OX*)YUcE>&>gHYliY7$5(O(&bp~<Pn=gDMzern7><nt2IUvb?DHDXHUIT9h@covw@gm54bp|J6vRu+Y79_;Iz4L(9ibWC6mCUrnSk=zzLxu}fLGGe3y(nMkR6jc<kGmpQP)LShB%Am+F$)$FGpLFU(k=e#NKFOk^r}8m4MAHE=Vv@;{>c>SVCn5SzCU{CVma1C-V5FPkayLhd(M5o`XniA|jD8d-5bx7fa#;xVO&g%%euQ;W(rFOJYjx2Vh^FKh<^XC1!-`0ZpLQ+jd+VZXAMOx3TCnb}^02@v3=bM57DuBo{!yJRu!91p3hsj3(W{qAYr^DePjotNK>nS+JZF1rBHYrDEA8zb`FF3#OE*YV-8j-zN2u9kI_A+9q$O4sd=hUfaFpu`r1@f(9|V^pIH9WwSW+?RZVq0<zX<^w6))vBc(8%mm88Turp(9lGnQAY<V!ccFcM{a!Li-`5>{BW`2^Z4b!`Q|5)(VcsN`2__=}P~JGAcEqQ|8katKpAp8|42e;*(b$|B#862DI*Z$>r)l;Iq2u~=u%pLsCRg^F414LUKyiW$KvQ94xv7KjPegQNCmK2|;mcD4^{<1_ZKXAjyKV=On(vwx`x0fIN-*A`8*?a>(M5z|(i9RSCLluRo9p}3ii{7L(gAJQ~?(lmcrL!R#cz8U4SsaLPcwT<&1$Cc8_Mx}VF$ImU|;2TH}%_*z($$SEyz5`XAMymAf8T<_$H(7)`<_7k293tKaZQS^dFj%@X8?d#8_jGKWz}z_?3@-696`FvxSl)(w6nB604gKp_%D0#VYcgQWu}tDGAtyb5b%kUREdMViUz%VZtErO3Y61NFi~g&%y)sSE7v%CwD_5R_zaG`mrMdQR;`~pp!GCN0iCB0*&)=e1Dxn2}j)E<lCE@ye)JcpanwT?fr7(vR%Iqw`N2krxfMk_R@>U9Jz<9>4%4JaTo0g&bVp&qT@tdmp=SME;Vc&ZR>0BA{=cJKkg;!_L3c7f%_U`mO$hmJPHO~Axd)%v2@io1zl#h#5Cgf3Bk=fGeNVuX*o`eCqgfe)~Z^vV;Dv`<j75`h`$e@WasHvt!mmJ@NB8{l-&p)At8=Q*^4jsG_NphRtjZ=Vp_h=n^)^9{l(JS-|CfY=7w&_^tN~mlUD6fT1X>`$iD531qaEuj4FuFHeJpxBG<apF&$3FkCo4}?eAF2sut5VDhp5b}ig+}1)D+`n`Kq<ID02~>_158I^3nn||MbP3!@DiMp<55agEyJ%BXciEhI5~Hmk>mB$E+GH=tOEu=iW;&TJ^xLsQS=ZX<m$jDTFV?=f)_+D(cKR4!D&an_@P2!aRECa9ko&T$$d6-oKP`%<W+W#bzAjq=(^(>5XMg(Hb^~-cOL4pmd-@2nUu+u3Iqzh&q^t2LrrTi#rGd-ZgwMKwv5Trm;mcF=Sfvx5}I&dvB`bu2m|$(#s&-xvsWDPvD9fOzLK~d36FxtaydYK)~^ul>Z<CLFqq=XmPr7z48e%U$7*26<^+JRuo?%2*a6_GJ9d}ko`x7?0HfB2sOTfntey=BkrA=ML>r_dU9_QR{nkD?%8i)P(bKcIgJ8D+7;0n9wmq0yOg2%~wp57Jv^0t@l)tPQCZ5S_hf8TJl1To5FW@~3=2KSNS0R0VFdq<%5v2KTE&rQjCMmxa>9gcpN-3HgH(zu<c^NnVdTA=+`M~q!Rk16A57X6nL^I^yAVu$nUSbR)kWMy;BWB`>x|>FB*%ktQR%7Av{!8TrGCk!5o(94%RF(8vzyO)eaZSAdkahxg8#r#N4v5wX5gvgV0bwk;z=Y-B|MyF^B0JBwYhrAY>3W{z3B@vHF?+o|H(fSaB10S(g+(ebHgKkV|EbS=V~n38jj!Qf{K_9k=+7g)`H<A~S67&Dw_K^2fySOM3Yl0ElGjK;FuAm{apEH<lAIK49xTrgspR|sgs@dM8Unc{$Jrt|<syOeLEJn$JLY1d<^woLa5=hMJ)f#leA1P*^keW(^VOON?V#V6{BIBlLywdXC;*B?OU+)}L%>2pr=$WK7&oemD`+UU1A2Jnb~9T{3r$bk2RkYx5I+K9XZ^Z|q46!B{V}Ax4>CRF1giZv3ID?OP^bh+ABqAUnS(O>oaBJL!;!|CPF+?gMUsk+v%#F3(U~jwk_tWqIRMyaCcB1XHj+WEN^F|@wyo?=FZY>~o79eMiU<x_k;2qx*diuystEa?g-S9<mpG_V9FZ174+`r%+5qjfQW2?RaV<tnDzW3>&X-h!FT|K}hF0MxV-+2g*Ze9Bin6i{A{SlZoJR+Kf0O(w#R$8J5i#(KiM>VQ5kJuwzn1{YB8L?DKE9Qfo24lUvXt<ABtbe7KO2eEaU~~e)*r>^*~8`J;W`M`6OiD5C7LE5muap9&~mV_xG8`hRp(}U^&L>a8?7M-WIuzuSHpR~dCGP}C(pCli1Kbp(Q#j<T%dY=Ugy%#S<-XLd9M5XQqO*sn_Sk`Vd|G4^NY^z4e<S-Kh)&p68oD_#$-W<24h%4rSuKXspM{+IIaG`<}iQi(v=!P1B^H5oOAm9qExpl`z6fp&6gY%lp69CRq&A|R`#1^*-H3RHZFIvq)zn<`bm>jV=REL^S%o|0`lqS(a}sVk8B_zGUQ<Q7lGq|2=l@tS(_zB<OS<2mcBt3A&QGzaT6sWCAqQ|#oMd3Dzw9YTFZ~gS8J9lC#9Qw<%d)|q7FhYjo36mK^7n-36RLk&C_y6IG6_?b4q&UO+Q1DrytSxxQzEP3Ga~<a#((|qhI~Ru6&px4fCWwNOdubH%)PFMS>?{zJ8*f;Cf}*&vM%PP6p?NOr&P_U7}OzV7~}cYc5B>vmEVsmx;(0)mWX7+0VfYg>$PjHOa#>MEYfA*@})D^1`KGM$nNg{Q8h?JZtr&gxk`__CikLQOa0J5lA%SiOFuW=Qnl8F)+MhE|iQL+VOJyaZ|@9oxS`f*$hF*G;n)Dy=iV`zHR9OF~VSu?lzOg%U(;EhItyy+BSnL8eoXeNq3(KOy^b#1uMzY2<O;0B)H6N#H3Y$=!jVrl&hYWMZg4Cl?_VZV)9d(ORfCEy6C<;6DF}p^}oG5kd+7Nz<C02Ke@TiK<ZgoJ^bfq*~~S0^$>;UR`EDx0r_+x-C{?%J}6^GE+%n?5oY#&rkd|mVVoYS_z6P3RV`rkoOr1;-n?bG!yKHX6H>1c>D9t{@#OxhsKmQd9w42I@j;hYm&Z+rmk;$f)?<!LEk1ok>4-MnTv$K|ej2`%uolg=$SOzl3w$2$`P@y0c!S`Ve!rx^oXGCaK`+n?ngOb#36`HBSRR*#uMKVSOA4pyoRo1sNEatv;}<=P7_uJ|OC_h=4KbeQ)US2dQNr7$ykIa*LIosbu~eK>sG1P0k@Mei6^rD{8yNU<%&>?aWm2ZI<i_4^GLri4yP5AWAUks8I?~>aW-d5Z$}t=2xe#MEa`JIj#BGF<<aNB9bUYT3sUc8-_of${&P*-;M=manT`L!Nd}S#K$m>&dB;NVc8Vr>0VJ}ecTW+9u5#Toxm9}y$Q$6*H{RG%>G;6xBDqE<BnvN5ZwlMT)1|3Rdf~e<AYSaH66BVuY<~Qv`%Aq4Hd~_MN!xw7QJAS1vYh+lH9CKzp;`GmL8_e5-sTgI)SA(c1T9U5>k`TvUE+>*9NQ%f1n9v|gqt<NLM5Q?#%v-KAL044RT)CnZqu^&AhU?-z3)TRyhFNalwMm4Q<9xNaiP|~dw@A)MayQ9IiWPA?fJ}J?k2UggL8uW`496RV()98&;ou|wngtL6ui^1W6Y9#u^P*-dKN3lWnEV`M_^T3@#9AY=WTfxK7_@koI-z`)(kw^ZWn!s=+<77t5?A*Z>4gJcTT}1lbkLxZt6Ra2`qgn1Poq4^Ns!#K5xp1_g##}FnZm_`<D@J}z^<=?VWwNoDTHBEPIJNuC}=J&$0n~9^^b|&tqjRWCMJ_*^uin=dI+Zk(d$pl`~kl;B#m$|iok>&WlM}FaZJYKf|htW%N0N-af9DRkDRyuN%VhoadLn8PxF|vMk7WYp%Pf_Zt!hH7~J_FkV&P?ehtctN^}__Z;4sSX<KQL8*zC=rJX%UO4<}fk7PAH&%~fGzttdliErdE4Q=`&m8tSNCZ}mcrdiKrROwZb0duaio)O~-89^vqIOc;9YaLOn@+8!f^vY9-Mr6-=SRezLntVo9g(SqU($>USK|R5o8>&DjJ;u#Y$O0;emv=8RRVbYMdq^Mxuf^qYqhvW8P&QTvi$Zl>xLG2^ye~~F7FORI0uL0f;eWFium#M3>tMIUgL1=!(}|%1?LX<hbaE1->03Q<p;-&-e`AhOhNRf_dl1&mL{jocTmB;#t@dwsM!gM89RDfXlQp38f6xK6NJQv3tlWZM{I--1re!jjACxO)bc>@hG4X*nE>1oh>IqlZ!Ebb69pof$IN?mK+My*!slfn`aU*o;lll;JjW>+WF3=)^L*$2O@ajsJGj9Fc08b=v-Q3T9_DzlT;^`VAvT-zhqri!D`8JuCxJ8>ROQdKC+<89psX-}>Ip}nKnM}A4^RIzJvRvewfg~(2Q8XZgGrtA?U1=gH^u0EGq0etd5(E8~PWUo9F{6IjZFU|HzyGw=C_HLi1@)Q3yXsTVVFi6+bX<?Rh5r&f>QfjL^@{s5i7@<}8k5h}J3=w%1;XdbfCeh{?^7i+?WwnJ2sXroyKwFx1Qtj~t8(_PzT=&>$Zy`}?fKR@-sWRH+=FP&p4AOt@*AcXTLnLqw`g7dCMPdi*@PO+EjJTQYP1r_%6!Mv-G2wgBr?W~TcnNp1iQ%(i3-0-U*Cz(SV3t@52q<?HIc;YpOs>bq!C9pt6UgkD1eHtm^Sup<!rFe0;N)PJGl>TP!?bL$f2RjC0Fz~-fSqAa&!4jjqM`Xo1*?}vXrx^hy6a4n};Z=Lt?@AN@GzE98DWRnui!5L08zJ&YC8D_t)k!>iUYK`~_B|`nwwXi0c4;*`f7AwjJeGw;YwAgb#?YMP*|RR%_vl)&1*vMYV`YOBVuzo)y0gMd_Z670*IV1!8`TXm03Y<AB|l{}LWdv}ysZ_$LHYAtizE6;x)ZrD1x9loTH)I(#FoZ)yFh<dav;N5ZaPC~4(Ydy!ve<%SzcHy^be1r160_CtN4m!CBwW#A8Wkyj0;+?TWk%7@Py5f=#rb@`iy5zd$&vlq~Z#z%NExRz_yCU4T83e6&EbCxFVsTQw9jjwKB(Iu;|1WqyYkSXL>;IM;M2nsV6g~!)%TzgKT6r9+6*9cgEE((({N0iA=&yoe5-%Y`&!esYE;FV{16GS|P=5oJGmBFQMFzIMyhrNrK1ng)qF6J(vOYy7bBkj1zFUj2ujUYVXF^$7Bhs-}!m5H2&p}u%&B!l=YW`y%C6g+=!@Wn~Spc4;OlJ8?S>e>mcSXIslrV&!vtc9)&u_m64*G2IAswLy$23}SzIbp?`*P~7y#I=RBoc#^%4T-bI$}eZ5fivnS!k2T4<2c>M9R?onD6-;WcG_qZD|^nrnOy`iL_JOdC&j=!QE?bCBp1WK=LU%4s0(K&U@N08PXGvKUO?dA6FbW8cQvsYb4os@*&r&w%W)^<ey*wKj|-WtZT(2IC>06QScX$(8R=va?GfB1#KWabpUlh_EM;T?jM5|+_SC;&=eAh6zGuqmBm#mhJb*-ekzFUwrA3F!lrrS^oJf%#+d04c@3^ujb}pAV+%uSs2y=W#GL~;-lK-S!*+EBm19GdtyK~X2>UVZ`4$_?X0*@gn2U_A6QnTNq6Hpy0HvBQ%`(q%_&y!$u!7#Xhjt-)Udq63cl#2nYYYoPpow)zkeX3?)vc{!N@RA1bgFy}k&###XDlzUo8jmnJbprb(&REbwI>3}SYkCu$18ko50@{rNyxpTNh{o0UGpSHNjx@#RpCSBSa))3^u)n^&B)g_$vG&4wkHf^o*Task5Eat1vj7}0zuZ++^(Lt5P)bactFtYIr(0jWkD#@g+aPdQJXj)*+Kt0~b>?^ue+(fp?bn$#Nbnd*`P3NzZNUD9NB<Qc12o59$DtgJtpCUbb+a-7MtkG1cAZ92;ze|73T@O;jzT%wqbd?(#chv~`-_tQ%vFH_k<+w!{G(Q^st!j<iMW}7Z%s_bMCIOydnTwT-?!|5u``u*LiJ^28yP12|I}d-zy$t^5=@;webZ}^v-c?XKZ<iK`c1p{Y`{Sqy(TEv6fCyM*u^X)Y;V>;Otdv`EDutnNESH#VghOS0OuEUqQ?UaJBv?;lN^-aq<wyfsPt_8NJtRyRT3S%-fZd&Qtl8NO2D_I01-3dq%Q{KiZ^Gan_{t2x;Zne#o|e|o^E`?v6*XEz_5>)g+VR?O`^Z2z|3q!Co^|80lcsbRyb)iF6CP=GAyuiX!tSd!-*SQOrNl5&?$78$*pjwHTHIlSs)S-z`)Uwk4fq^dW?)THbHIfC+$u@gO2zKfEzJRxw{HeLU?r{`Jt^qm|}cY0<`@72E(JlIOQB-q#l*7@hTn$Kkg%OZ&uUSA;>j?mvW{HY;RT4Hda*wUgx}715!<OEN1|;p)xwY9;9+IZ~0BjyTgq!Uizcqoz1Reji&52NZ1M4T?m`BKWG83Y+<Kt;^Ay*4qCGTiBX=0$!}8;PD>4#G6ZziU<h)&F{`ZdJg)UYm)PLI$6)GN_Seu?oIoN97eWC)1g)N==wXO5GtjJszBCLUPWVewIseen$7ivXV)g#IXy)%vN;PtM)(tk&<mx)7rAqM;O>}at&k+-fucj9Mmh(lv+jZ?O&7PrkExJ2`40bkf^uOzv!S$>r8<(f-d}{g#U#4JQvMhs3A{635Zu-eocK~cazxv&*{_Z=YCx6L*e3Cp;as)_D-^uSdeXHT{K}b)_d*cB}>^jQsyArlSUiCC4@?1Y+fgy~`e{%=t>?Wdu?!K!-o;%Z?h67HGCz#&`JEhO3IvjaCt&`O{LX?y&wA`-^`^1Jr5~W;%jhh?2?Lk|aIkFF{9rI;-Hyd`R=!*n83{aN;6*JOjjbi6*H<uV#!)DvkCRu4nG#+DQ&q2nvbI{EsLj-e*3##~~)xv8&nph)jP8fOu*W?^bX$r4cdqj-5iH<@2#N=6Cj&&L4an2Bim)oOwd9)UB24OMpyc|3v-~hyK-|Z&@G%<jFY@5Z`tQ`^=xj1FwRtN{7LW%Mjk(~3n5XskJZpFGxhTIPpQvX@Wz%Js-J7FV~QUK_J>?sR!pUb&*W)f(SdIlfKD$q!N4qx$+Zz$OZ{u21UcAo?HjlzEgaNjl{ki)FWU!YXB9`K3|vl09yor<$Ph-Sz_w)lnm!WzK=kh(U?4;U0Fu7djR$EO{!dSR<oFg|WT@<(W>){Rfi8X?Vb)NJvUy+SoqDE5euAtG!kh?)r4;$jMZKAE3RrSbdkZarj4h?!siSQFkzxjj^67E&VV;4~?RTyJ)CInGO$_n1jcj19%Y<!BeiZhMJI3f;`v)`2P#-m`eK<iv7ZlN-Z9gsZDB0IeO|JxzMcB9oy^_#K;=KE0s=!ISs|XRS^L1#bVd|3(xMmuHIt{F$g`+GEYxD9=-o@Vkb=>^Q=PGOR1l;&aH%;};8?e-y(ux}C`W{cPPiHaBhaWhhTQ-CrMwQK)KN2(k*+vtZw{==2>|GvdQOD^oymV>oLmM^YCFnG4xZEcS{y*UM_Lx67du+RqjC>k%_nxt`=G7P?(2@brrKbs4goR@5iU2zgk2>Puy;aEvlmyE+-MUQ032Cg{Y}rvNJ3g>zw|z7T3t#+JR14>u<WvEraQcajEj;yGjy)2;<e8O||`TDErg#2rrTlj%+D7R<5_I-#HKn5ri^SFLVP+@3*tKKdz{9ii5)iORq7ak-*YV){!7fLd!$R{hEq#OsaVWw<8O?wZy6<LYg`Pz(WJUpK_p7!WWJ>a$rqJuMb$d}w}kb(9SSZ3+vft0e|@XKJ`kw>K533<LD4F?c_9@m0@KRUPXhjRORoK)^<bf5fhMTPa;*=>1hWfASG`+e#vq$i5(DdCjaO-)P1)GE(G_DFV;&oi2jm8+fyd+~ee$9J4{#Q;3$)>?x&$a*(;;3SbFcvs%KRQf))?i3qdLC&XmIZ!uZPuRMS7pOEK|7(&?9M{qZJDCFX^5IcFJE*M@TxA4<jhBs%XJdg9^8KGQ|Sg}l25Tl}WM>{qpQ)FbwywUxPukuy2eqK<kF16RF$I?w>LzJR6SCvX`Ahf_@@Ch)#|0`kfNfAVltN+wR25ddMg1RL_A<s6(H@QO-Jx|~y%P)Y@N2C^c<Njh4k3DXVUHXh}Ttf_pX~Ge)<Ft?rXCqX>kxl8NAvo(9-Ll7y5oxpI4D8&eY9Tm_g-S&gP8?c}Y|k93FM=`UP0Tw<787-3iHt?ZbCp!2hWk|oaarD*iE)Gj8w0Fi3+7T@GDlB}$jKdUGo-)sNKYsWee$*z690mw$ibugVMHQS07PWN7#JVCLkzF-P}m({yr<#bF!wywv+r9K{>W{o)1i`)etrY`$~yndrP901=m-mcp$Nda+*Ue*gQVcCXIomw9S=za*Z>j0KLT?~9{viy6m`GHFT2AKzYJZ>gsihJu0IMAM|Uw>XkuWv%C$tC0NV9IW20Sn*{qWY+#UKCVo|&mVS%lk;3w9LAfDjJ0gs%JCn&BP@CT0&20f12Io2;>X-2bi(oo;|;!OC1uthWQTYR}wGpngW`TU1Z0OI{%Jcrk5@eaEx7v54ogi7OpR)EG2ytGe4+nUvSojD%enx67!YJjIo=B5rCe=uDrzJj3e<>5glmsN*|iJRliC>!J_(hVwITWTe~m3~%<P0co&8kR2{{ayB`U-aAxp7CDG1>o&aT6j?Yk6dPF_4iirUHH~g9W151Y_XTza)iq;_j6GXB*#WpI+yu!72|pBN{(vOV?_A|9Muc{=djF<U#tFCNjAq1H`0f&=*=~2JDk~8R&4Vt3H{3&|IBTqGU>0%_CNrIvO&2|X1gIs93b<Rf;9rESZ2H#J`>$tKElE;kFf9yBP@J#goR%oVd2XW?y(W#9@GBiL4IS5oX@ns9-`0P6YZ+ezDWlA*}fV9fP3HuaxLx6(||G3Zvzu4GW8f<#3G#Fd7!|roe@xL`knnwFtEV!AEW-zffid_adL<T1TY=AHD||(wk$sfNO7b~AhAn9A%DY@xFx>BMGQL<11ZV(8-HC3jD+^FnLv|4jZ!|NIgT`^#uo5#GP4eU2vTMr(d1G!UVS~m;QK@f-A#+i_)4Kl9}u@Iv}ZKqIgtudsza|(?aE&LGsiDjNbnaMq9@5s{)Ug1Z{=#QB3n8CtSn%^Kat}87uW{2f=&29PU0cEI4gDszA2#MhtVpLns9RBlXSp2Sr`a4k7jnH)|eO~S=J8Mjrv!Ko)<?LxvW`ks|V-c)|4IwgK>*{DYVBle&&v#0YWGz9Az{Sm(mMoF2ogO|44lvvNw36Et9R&?4KBoj69M?K4cc4tb3?Fcq7bJi-q#dNz8}ATV1Zw_>UaCQA;<nI4xvKmaV|ubXI$_T@nKH(#=Eus&rzOu5Yi3*Z#h~<|j&_8kv|HC&h>2Rrwa{n3&f!e-*B^CDjTglVB<`^QK%wiQ=kMfLT5X*FTWz-`^aU?{7|uCuX@~9^apw6>H{0seBdN-EFC2zHoP6s+OqLS|xu{ET5iA<-1IJ>T+pyPpWx$592&7L6aCDXP#DyMU-!fF}0ZpM{Vz}9yC@48Rr(Y<H~4x>r{V7s;@{>X8Gpohg2y6jRUQ9cauS7x%<dU&FXEj0AnbWYCp`=eCaA^7LE3P*_*P7_jr}6HJr_}e2Jr!JU-y?eU>&IJnRmG8*sa6LqE%yO5)vahThT16Q9^G#aTx`+cLwK>2|jad`)N6H;=r2v;4p$zYokY;uxjYVwM<w;0QIkZAzyND70;$FdGUe^j*d<g>iSsK!H}#>%iMx1Lajz!#b$g!T5&mUZ}BsGi33cOz2vy?!JK^smE}3nTeqiE2mGcJzy7u`urOGnfT98D+gwCj7p`p$y3~R4*<p?yHnK9=7fIN=q>AXmgx4$y@~k5>skKeV=b^8Aqq5>sGF;%0xuO`?IbULrh9_Vx*;?LtEz<-#96U?U944pu*yOcu(EcoSh+3%pJBy)L)5C`h4W(J^0r)pE@CUCLKQ2b+m9jn2pHXW$x-Gv?G$DJNWeb?(GJP%*@fzZhB_{*=?q}z`F;WAWgV@0f|6AAilSaQdkKv_?k@PF;j8UY!9l-#i?+W5g%tNJCBh>9ViQcH<T)c42~b4h%^;C9J_m`1p`r-t%?G8q^^X3t|6%_0K49m9??KZrcwhOxaw2@zg6XBpH^7CfwJMuL1i*#kP6Ug-`c}napvVPe_z#V!j&@NGziNDd*eJ59|Ln*rUlT?M>p{V0UC{{#`6WR34DJtKJ?_(dp;oSx@>h{jcUIQn@=n<=%LJj7#q+pt^9_+zVAkN3H6AM^Xu!{;DJ@=dsaRQ0xdg3xe~pp{tB0qV?^Oe)aZ?LcA@@;HS70pHcqn0hdDfipzxl+Y@(gfT<&KoJg(W}p*u?GF_qz0hW%&=oU#Aq9%YULmnbW48^?w?AmuaWz54XmHc<6L&4_#eX<;xGkm}AoB`yW5eo-ai)=KUIY|3>Kq=_xO&N#zPL<qE67E4q0-OPBho`g+Eo;F}_r8X<`^0ysYJ{$G({U!MSPfz=E$f%ugYKM=i2<SKWINycrBD@o@%e^WXwRt52hnP&4YvA;^WRxXsUQp2{Mlzk-mjMj=zHL?4g&j88>+hhi|Z<MF6=z|DnGW|F0VQ1HOwEF-4*Z-@PxMO$A-Z+~%AQC8vEDLZT8V_*Nk@yT9_>cbk?+<Kv+24QH{_Ce0>wnQ_8Ic`*Lpw`rs`QuDe*gVvH~sPZ?+6PDWHd<*WM4DE*DXyIwD57@v?NelL19HN%)V|1Uw6WwB$<sB-U$lts#ZDneK+{Nr%7@g`??o=4Jn1V6vv9~2gMFFE{19RE*1QKK;J{C)PD<HDSXpOz+IsD*HVOs`kh!s*Ab%+u`6}{u>teyOB5g4hW0a%CO-a4<3Kd1oYj7MfuaFcTddZBCym^+^bf6$6-XKQVNhkJhWKdwBN>pYww}}KI$i@?#rx-b%chGLW1>a?MkjKR`jRR|7BF4j*k<jrFd!zra(ZD$k`zakv=Q<hDZIw?|1r{2FJrtQ9{ZGB#}|tCJOLDUrC6=L)wRvp1ScrY@qC4gEc5sY%(8DgyTVPD153R-6Q0R@M7i2wC5IS@!YK@87n8?lsC(8bS^RThJ0whhCM<s@zpa*WqLNIUu=0sG@`=RpsZ<dX7ve%PVr}b<z_lBuiVWJ%&g2ySCfMwfFE|=~VPeUf*e_PPHBXACSE$1vSMYb$H4eRUNCC<wUKP9pk4h{S$$2Y<e!X+&`-x$>Ei9bFe*zZ{@x5BUuM~=YA@-OG%5u7AgZf*dMk=$>$aepp;1G)KvRrF0um=>!?}=bYKPWP|Pz#v|l?mS_%rB}G75<sa1SLd{0OmDI{#OKpyKGGtxgljg^rL5WZ(_{_SCOt>AzFe$JSk<WGebS4m%4SwEg+%W@pw>N*sU2zk9b9_DPlV1o*Bc5XeS3v6M5MSiN0p&9bqq0-mE!vL_GtJchgY!)dM5c<6BI)cb4aSHo!JAi(D}(T?-6f{mgtT?r<4ez<Pnw-sIagtVs@D(Y0`?yxX0j)NqSWZo+q2h4p2c?QshPJCxY)Q0|&^&tg!wV$Dy2^5^zoaw>{)iGM<wjV<0rLf_cH@YM|rU)2qEuFL6|D;2g2ID+Uat4X1%8>Bt>wdx=nbZyi-j2)QzTte}lHuMP-3v?8GtpOAyD-ZlDLOJxL!EH*+0JO3p!2AX<6WyR2{wVN6RHQ9S-kF47P7D}MejE&DTM$r?IL8?A{^qzCM1^iXAdi}l<;o@O6VT}$J(>Ij{dj-~z=9Yu$>~5F8+4qQ;zo}d*E`Na8+Vw!%^l<Y7^zRNOBZ_m+Jy9-i9D&F)pol;bCVR&90Ku(pt*Z2I?lnXlphu?Im^3<Eb^0T!!d^?GFBs~4ToaWb6W;_U9;<3=qwyudXx6q3bn&`iz8Zx-;4fY0Y7wRi$pjF{SPi2ODM5H^(-GB0BK0Wg-2FgjU?ht`jeSM9xSy$4o3k5ex_4U8Hs)W1f{WY@&<wNS?G6h?2io>E4Q@zx^?yisK%6|*2y!&AFE>1#Y)(jI3K;_ikJvCJRSE>gzjQcuL~<cYI2ju1ylCxR4Ro!w3*iv(ca&{M|7;r=T~J(i~)rilN>LJn@X;pz^uwHwvsf0X=ZyTrJQ4g6Pm$}ej*<jf8rUO046_Jg&@PhPr36S%5Y->=AzO8zLFDZ@f}h@NmgK<oD_A#jkFSjuK+j1ktItZM&#`6#4RmZH*zGQP~7(O>v$9=${^l7%R0dBR5GHtr3<V693CT?ltO4!ox9x}i&Yd+vUJlrN3)MS*QG_|hj2w>xTGL1KOEe}Gj{?%g>zLkj7;+5JJTocw8_viN7Tnrx~k0zCX9vaemPfFY%*r1RIvZo&Kbwp|FIbhq@i!HCuX`-Vqp!w<#Z&L&6SCP%dy0Ug^G+Tk?X#0un{{EE4Poh4T5E#is~BzeBaYMtqEuw*wBafNTdGdwYCOrn<gIdOcPo*O%%ILljoUd`6XD^qvFEB3|b(R!%qFb0mv9ZIs'''
def _load_v45():
    src=zlib.decompress(base64.b85decode(_V45_PAYLOAD.encode())).decode()
    mod=types.ModuleType('v47_embedded_v45')
    mod.__file__='<embedded_v45>'
    mod.__name__='v47_embedded_v45'
    sys.modules[mod.__name__]=mod
    exec(compile(src,mod.__file__,'exec'),mod.__dict__)
    return mod
V45=_load_v45()

# Reused validated substrate functions/classes.
surface_context=V45.surface_context
raw_rep=V45.raw_rep
IdentitySet=V45.IdentitySet
PairVerifier=V45.PairVerifier
OjaSubspace=V45.OjaSubspace
V39=V45.V39
CAND_A=V45.CAND_A
psoft=V45.psoft
psoft_batch=V45.psoft_batch
make_persistent_stream=V45.make_persistent_stream


def sha_bytes(b:bytes)->str:return hashlib.sha256(b).hexdigest()
def sha_json(x)->str:return sha_bytes(json.dumps(x,sort_keys=True,separators=(',',':'),default=str).encode())
def source_sha()->str:return sha_bytes(SOURCE.read_bytes())

def _json_safe(obj):
    if isinstance(obj,np.ndarray):return obj.tolist()
    if isinstance(obj,np.generic):return obj.item()
    if isinstance(obj,dict):return {k:_json_safe(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)):return [_json_safe(v) for v in obj]
    return obj

def _float_vector(value):
    """Read either new JSON arrays or legacy NumPy string renderings."""
    if value is None:return []
    if isinstance(value,str):
        q=value.strip().strip('[]').replace(',',' ')
        if not q:return []
        a=np.fromstring(q,sep=' ',dtype=np.float64)
        return a.tolist()
    try:return np.asarray(value,dtype=np.float64).reshape(-1).tolist()
    except Exception:return []

def atomic_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(_json_safe(obj),indent=1));os.replace(tmp,path)

def atomic_gzip_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    with gzip.open(tmp,'wt') as f:json.dump(obj,f,default=str,separators=(',',':'))
    os.replace(tmp,path)

def load_gzip_json(path):
    with gzip.open(path,'rt') as f:return json.load(f)

def atomic_pickle(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    with open(tmp,'wb') as f:pickle.dump(obj,f,protocol=5)
    os.replace(tmp,path)

def load_pickle(path):
    with open(path,'rb') as f:return pickle.load(f)

def interval(vals):
    good=[float(x) for x in vals if x is not None and math.isfinite(float(x))]
    n=len(good);out={'n':n,'mean':float(np.mean(good)) if good else None,'low':None,'high':None}
    if n>=2:
        sd=float(np.std(good,ddof=1));crit=float(student_t.ppf(.975,n-1)) if student_t is not None else 1.96
        half=crit*sd/math.sqrt(n);out.update(sd=sd,low=out['mean']-half,high=out['mean']+half)
    return out

def current_rss_bytes():
    x=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(x if sys.platform=='darwin' else x*1024)


def short_bin(s):
    if s<5:return '4-5'
    if s<6:return '5-6'
    if s<7:return '6-7'
    return '7+'


def normalize(v):
    v=np.asarray(v,dtype=np.float64);n=float(np.linalg.norm(v));return v/n if n>1e-12 else v.copy()


# -----------------------------------------------------------------------------
# Development identities and Stage P
# -----------------------------------------------------------------------------

def make_ids(seed,n,exclude=()):
    rr=np.random.default_rng(seed)
    alphabet=np.frombuffer(b'0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',dtype=np.uint8)
    used=set(exclude);keys=[]
    while len(keys)<n:
        k=bytes(rr.choice(alphabet,size=6))
        if k not in used:used.add(k);keys.append(k)
    targets=rr.integers(0,256,size=n,dtype=np.int64)
    # Keep Candidate-C's small within-identity conflict population so Stage N can
    # report whether negative events include genuine context-dependent conflicts.
    nc=max(1,round(.125*n));conflicts=set(map(int,rr.choice(n,size=nc,replace=False)))
    return IdentitySet(keys,np.asarray(targets,dtype=np.int64),conflicts)

def dev_target(ids,i,w):
    y=int(ids.targets[int(i)])
    if int(i) in ids.conflicts and int(w)==3:
        return int((y+1+(int(i)%251))%256)
    return y

def positive_events(seed,ids,smoke=False):
    rr=np.random.default_rng(seed+470101);reps=1 if smoke else POS_REPEATS
    ev=[]
    for _ in range(reps):
        block=[(i,w) for i in range(len(ids.keys)) for w in POS_WRAPPERS];rr.shuffle(block);ev.extend(block)
    return [(surface_context(ids.keys[i],w),dev_target(ids,i,w),i,w) for i,w in ev]

def learn_positive(seed,ids,smoke=False):
    verifier=PairVerifier(cap=128 if smoke else POS_CAP,tau=POS_TAU,topk=POS_TOPK)
    learner=OjaSubspace(seed+470003,rank=POS_RANK,eta=POS_ETA)
    ctxmeta={surface_context(ids.keys[i],w):(i,w) for i in range(len(ids.keys)) for w in POS_WRAPPERS}
    audit=[]
    for c,y,i,w in positive_events(seed,ids,smoke):
        p=verifier.lookup(c);found=verifier.observe(p,y)
        if found:
            source,cur,yy,sim=found;learner.update(raw_rep(cur)-raw_rep(source))
            a,b=ctxmeta.get(source),ctxmeta.get(cur)
            audit.append({'same_identity':bool(a and b and a[0]==b[0]),'source_id':None if a is None else int(a[0]),
                          'current_id':None if b is None else int(b[0]),'sim':float(sim),'outcome':int(yy)})
    U=learner.freeze();false=None if not audit else 1.-float(np.mean([a['same_identity'] for a in audit]))
    return U,audit,false


# -----------------------------------------------------------------------------
# Stage N: short predictor and bounded negative-collision verifier
# -----------------------------------------------------------------------------

class ShortPredictor:
    def __init__(self):
        self.count=np.zeros((256,256),dtype=np.uint32);self.total=np.zeros(256,dtype=np.uint64)
    def probability(self,prev,y):
        return (float(self.count[int(prev),int(y)])+SHORT_ALPHA)/(float(self.total[int(prev)])+256.*SHORT_ALPHA)
    def surprise(self,prev,y):return -math.log2(max(self.probability(prev,y),1e-300))
    def observe(self,prev,y):
        self.count[int(prev),int(y)]+=1;self.total[int(prev)]+=1
    def bytes_used(self):return int(self.count.nbytes+self.total.nbytes)

class NegativeVerifier:
    def __init__(self,cap=NEG_CAP,tau=NEG_TAU,topk=NEG_TOPK):
        self.cap=int(cap);self.tau=float(tau);self.topk=int(topk);self.clock=0;self.serial=0
        self.traces=collections.OrderedDict();self._keys=None;self._mat=None;self.peak=0
    def _invalidate(self):self._keys=None;self._mat=None
    def lookup(self,c,z):
        c=bytes(c)[-12:]
        if not self.traces:return []
        if self._mat is None:
            self._keys=list(self.traces);self._mat=np.stack([self.traces[k]['z'] for k in self._keys])
        sims=self._mat@np.asarray(z);kk=min(self.topk,len(self._keys));ix=np.argpartition(-sims,kk-1)[:kk];ix=ix[np.argsort(-sims[ix])]
        out=[]
        for j in ix:
            sim=float(sims[j])
            if sim<self.tau:continue
            k=self._keys[int(j)];t=self.traces[k]
            out.append({'source':k,'sim':sim,'y':int(t['y']),'surprise':float(t['surprise']),
                        'time':int(t['time']),'eval_id':t.get('eval_id'),'eval_pos':t.get('eval_pos')})
        return out
    def observe(self,c,z,y,surprise,eval_id=None,eval_pos=None):
        self.clock+=1;c=bytes(c)[-12:]
        if c in self.traces:
            t=self.traces.pop(c);t.update(z=np.asarray(z).copy(),y=int(y),surprise=float(surprise),time=self.clock,
                                         eval_id=eval_id,eval_pos=eval_pos);self.traces[c]=t;self._invalidate();return
        if len(self.traces)>=self.cap:
            self.traces.popitem(last=False);self._invalidate()
        self.traces[c]={'z':np.asarray(z).copy(),'y':int(y),'surprise':float(surprise),'time':self.clock,'id':self.serial,
                        'eval_id':eval_id,'eval_pos':eval_pos};self.serial+=1;self.peak=max(self.peak,len(self.traces));self._invalidate()
    def capacity_bytes(self):
        # Learner-available semantic state, not Python overhead.
        return int(self.cap*(12+D*8+1+8+16)+64)


def pair_clean(U,c):
    z=raw_rep(c);coef=U@z;z=z-U.T@coef;return normalize(z)

def negative_records(seed,ids,smoke=False):
    rr=np.random.default_rng(seed+470301);reps=1 if smoke else NEG_REPEATS;rows=[]
    for _ in range(reps):
        block=[(i,w) for i in range(len(ids.keys)) for w in NEG_WRAPPERS];rr.shuffle(block)
        for i,w in block:
            y=dev_target(ids,i,w);raw=surface_context(ids.keys[i],w)+bytes([y])+b'\n'
            rows.append({'identity':int(i),'wrapper':int(w),'raw':raw})
    return rows

def eig_basis(K,rank):
    K=np.asarray(K,dtype=np.float64);K=(K+K.T)*.5
    vals,vecs=np.linalg.eigh(K);order=np.argsort(vals)[::-1];vals=vals[order];vecs=vecs[:,order]
    B=vecs[:,:rank].T.copy();return B,vals[:rank].copy()

def random_protect(seed):
    rr=np.random.default_rng(seed+470777);A=rr.normal(size=(POS_RANK,PROTECT_RANK));q,_=np.linalg.qr(A);return q[:,:PROTECT_RANK].T

def neg_coeff(U,current,source):
    d=raw_rep(current)-raw_rep(source);d=normalize(d);return U@d

def _choose_shuffled_sources(events,seed,Uplus):
    """Similarity/surprise-matched wrong-source control.

    Exact update count is preserved. For each genuine gated event, choose a wrong
    source from the same current/source surprise bins whenever possible. Matching
    prioritizes PAIR similarity; temporal distance and repeated-source reuse receive
    small penalties and are audited. This intentionally favors a harder control over
    exact source-multiset preservation.
    """
    zc=[pair_clean(Uplus,e['current_context']) for e in events];zs=[pair_clean(Uplus,e['source_context']) for e in events]
    out=[];use=collections.Counter()
    for j,e in enumerate(events):
        pools=[('surprise_matched',[k for k,x in enumerate(events) if k!=j and x['cur_bin']==e['cur_bin'] and x['src_bin']==e['src_bin'] and int(x['source_y'])!=int(e['current_y'])]),
               ('source_surprise_matched',[k for k,x in enumerate(events) if k!=j and x['src_bin']==e['src_bin'] and int(x['source_y'])!=int(e['current_y'])]),
               ('global',[k for k,x in enumerate(events) if k!=j and int(x['source_y'])!=int(e['current_y'])])]
        chosen=None;best=None;level='global'
        for lev,cands in pools:
            if not cands:continue
            scored=[]
            for k in cands:
                sim=float(zc[j]@zs[k]);bd=abs(int(events[k]['block'])-int(e['block']))
                score=abs(sim-float(e['sim']))+0.002*min(bd,8)+0.0002*use[k]
                scored.append((score,use[k],hashlib.sha256(f'{seed}:{j}:{k}'.encode()).digest(),k,sim,bd))
            best=min(scored);chosen=int(best[3]);level=lev;break
        if chosen is None:chosen=j;sim=float(zc[j]@zs[j]);bd=0;level='self_fallback'
        else:sim=float(best[4]);bd=int(best[5])
        use[chosen]+=1;x=events[chosen]
        out.append({'source_context':x['source_context'],'source_y':int(x['source_y']),'source_surprise':float(x['source_surprise']),
                    'level':level,'source_event_index':chosen,'matched_similarity':sim,'block_distance':bd})
    return out

def learn_negative(seed,Uplus,ids,smoke=False):
    verifier=NegativeVerifier(cap=128 if smoke else NEG_CAP)
    short=ShortPredictor();K_all=np.zeros((POS_RANK,POS_RANK));K_gate=np.zeros_like(K_all)
    history=bytearray();events=[];all_conflicts=0;target_conflicts=0;gated=0;event_index=0
    rows=negative_records(seed,ids,smoke)
    for row in rows:
        for pos,y in enumerate(row['raw']):
            if len(history)>=12:
                c=bytes(history)[-12:];zp=pair_clean(Uplus,c);pre=verifier.lookup(c,zp);prev=int(history[-1]);sur=short.surprise(prev,int(y))
                chosen=None
                for cand in pre:
                    if int(cand['y'])!=int(y):chosen=cand;break
                if chosen is not None:
                    all_conflicts+=1;target_conflicts+=int(pos==12)
                    a=neg_coeff(Uplus,c,chosen['source']);K_all+=np.outer(a,a)
                    ev={'current_context':c,'source_context':chosen['source'],'current_y':int(y),'source_y':int(chosen['y']),
                        'current_surprise':float(sur),'source_surprise':float(chosen['surprise']),'sim':float(chosen['sim']),
                        'event_index':int(event_index),'block':int(event_index//256),'current_id':int(row['identity']),
                        'source_id':chosen.get('eval_id'),'current_pos':int(pos),'source_pos':chosen.get('eval_pos')}
                    ev['cur_bin']=short_bin(sur) if sur>=4 else '<4';ev['src_bin']=short_bin(chosen['surprise']) if chosen['surprise']>=4 else '<4'
                    isg=sur>=SHORT_SURPRISE and chosen['surprise']>=SHORT_SURPRISE;ev['gated']=bool(isg)
                    if isg:K_gate+=np.outer(a,a);gated+=1;events.append(ev)
                verifier.observe(c,zp,int(y),sur,eval_id=int(row['identity']),eval_pos=int(pos))
                short.observe(prev,int(y))
            history.append(int(y))
            if len(history)>12:del history[:-12]
            event_index+=1
    # Exact update-count matched shuffled negative evidence.
    shuf=_choose_shuffled_sources(events,seed,Uplus);K_shuf=np.zeros_like(K_all);levels=collections.Counter();sdiff=[];sim_orig=[];sim_shuf=[];bdist=[]
    for ev,ss in zip(events,shuf):
        levels[ss['level']]+=1;a=neg_coeff(Uplus,ev['current_context'],ss['source_context']);K_shuf+=np.outer(a,a)
        sdiff.append(abs(float(ev['source_surprise'])-float(ss['source_surprise'])))
        sim_orig.append(float(ev['sim']))
        sim_shuf.append(float(ss.get('matched_similarity',pair_clean(Uplus,ev['current_context'])@pair_clean(Uplus,ss['source_context']))));bdist.append(int(ss.get('block_distance',0)))
    B_all,val_all=eig_basis(K_all,PROTECT_RANK);B_gate,val_gate=eig_basis(K_gate,PROTECT_RANK);B_shuf,val_shuf=eig_basis(K_shuf,PROTECT_RANK)
    return {'B_ALL':B_all,'B_GATE':B_gate,'B_SHUF':B_shuf,'eig_all':val_all,'eig_gate':val_gate,'eig_shuf':val_shuf,
            'all_conflicts':int(all_conflicts),'target_conflicts':int(target_conflicts),'gated_updates':int(gated),
            'verifier_peak':int(verifier.peak),'verifier_capacity_bytes':verifier.capacity_bytes(),
            'short_bytes':short.bytes_used(),'shuffle_levels':dict(levels),'shuffle_source_surprise_mae':float(np.mean(sdiff)) if sdiff else None,
            'shuffle_original_similarity_mean':float(np.mean(sim_orig)) if sim_orig else None,'shuffle_similarity_mean':float(np.mean(sim_shuf)) if sim_shuf else None,
            'shuffle_unique_source_fraction':float(len(set(ss['source_event_index'] for ss in shuf))/len(shuf)) if shuf else None,'shuffle_mean_block_distance':float(np.mean(bdist)) if bdist else None,
            'events':events}


def build_transforms(seed,Uplus,neg):
    Brand=random_protect(seed)
    return {
        'RAW':{'U':np.zeros_like(Uplus),'B':None},
        'PAIR20':{'U':Uplus,'B':None},
        'PAIR-RANDOM-PROTECT8':{'U':Uplus,'B':Brand},
        'SIGNED-NOGATE':{'U':Uplus,'B':neg['B_ALL']},
        'SIGNED-SHUFFLED-NEG':{'U':Uplus,'B':neg['B_SHUF']},
        'SIGNED-GATED':{'U':Uplus,'B':neg['B_GATE']},
    }

def apply_transform(z,U,B):
    if not np.any(U):return np.asarray(z,dtype=np.float64).copy()
    coef=U@z
    if B is None:
        rem=coef
    else:
        rem=coef-B.T@(B@coef)
    return normalize(np.asarray(z,dtype=np.float64)-U.T@rem)


def development_identity(seed,smoke=False):
    return sha_json({'version':VERSION,'kind':'development','seed':int(seed),'smoke':bool(smoke),'design':DESIGN_SHA256,
                     'pos_ids':32 if smoke else POS_IDENTITIES,'neg_ids':24 if smoke else NEG_IDENTITIES,
                     'pos_rank':POS_RANK,'protect_rank':PROTECT_RANK,'neg_tau':NEG_TAU,'surprise':SHORT_SURPRISE})

def learn_development(seed,smoke=False):
    pids=make_ids(seed+470011,32 if smoke else POS_IDENTITIES)
    U,audit,false=learn_positive(seed,pids,smoke)
    nids=make_ids(seed+470021,24 if smoke else NEG_IDENTITIES,exclude=pids.keys)
    neg=learn_negative(seed,U,nids,smoke)
    transforms=build_transforms(seed,U,neg)
    keys=list(pids.keys)+list(nids.keys)
    return {'seed':int(seed),'U_PLUS':U,'B_ALL':neg['B_ALL'],'B_GATE':neg['B_GATE'],'B_SHUF':neg['B_SHUF'],
            'B_RANDOM':transforms['PAIR-RANDOM-PROTECT8']['B'],'positive_pair_updates':len(audit),'positive_false_pair_rate':false,
            'positive_rank':int(np.linalg.matrix_rank(U)),'positive_orth_error':float(np.max(np.abs(U@U.T-np.eye(POS_RANK)))),
            'negative_all_conflicts':neg['all_conflicts'],'negative_target_conflicts':neg['target_conflicts'],
            'negative_gated_updates':neg['gated_updates'],'negative_eig_all':neg['eig_all'],'negative_eig_gate':neg['eig_gate'],
            'negative_eig_shuf':neg['eig_shuf'],'negative_verifier_peak':neg['verifier_peak'],
            'negative_verifier_capacity_bytes':neg['verifier_capacity_bytes'],'short_context_bytes':neg['short_bytes'],
            'shuffle_levels':neg['shuffle_levels'],'shuffle_source_surprise_mae':neg['shuffle_source_surprise_mae'],
            'shuffle_original_similarity_mean':neg['shuffle_original_similarity_mean'],'shuffle_similarity_mean':neg['shuffle_similarity_mean'],'shuffle_unique_source_fraction':neg['shuffle_unique_source_fraction'],'shuffle_mean_block_distance':neg['shuffle_mean_block_distance'],
            'negative_events':neg['events'],'development_keys':keys,
            'pos_keys':list(pids.keys),'neg_keys':list(nids.keys),
            'development_raw_bytes':int((32 if smoke else POS_IDENTITIES)*len(POS_WRAPPERS)*(1 if smoke else POS_REPEATS)*14
                                        +(24 if smoke else NEG_IDENTITIES)*len(NEG_WRAPPERS)*(1 if smoke else NEG_REPEATS)*14)}


def save_development(out_dir,dev,smoke=False):
    d=Path(out_dir)/'development';d.mkdir(parents=True,exist_ok=True);seed=int(dev['seed']);npz=d/f'S{seed}.npz';meta=d/f'S{seed}.json'
    tmp=npz.with_suffix('.npz.tmp')
    with open(tmp,'wb') as f:np.savez_compressed(f,U_PLUS=dev['U_PLUS'],B_ALL=dev['B_ALL'],B_GATE=dev['B_GATE'],B_SHUF=dev['B_SHUF'],B_RANDOM=dev['B_RANDOM'])
    os.replace(tmp,npz)
    md={k:v for k,v in dev.items() if k not in ('U_PLUS','B_ALL','B_GATE','B_SHUF','B_RANDOM','development_keys','pos_keys','neg_keys')}
    md['development_keys_b64']=[base64.b64encode(k).decode() for k in dev['development_keys']]
    md['pos_keys_b64']=[base64.b64encode(k).decode() for k in dev['pos_keys']]
    md['neg_keys_b64']=[base64.b64encode(k).decode() for k in dev['neg_keys']]
    md['development_identity']=development_identity(seed,smoke);md['npz_sha256']=sha_bytes(npz.read_bytes());atomic_json(meta,md);return md

def load_development(out_dir,seed):
    d=Path(out_dir)/'development';meta=json.loads((d/f'S{seed}.json').read_text());npz=d/f'S{seed}.npz'
    if sha_bytes(npz.read_bytes())!=meta['npz_sha256']:raise RuntimeError('development npz hash mismatch')
    q=np.load(npz);arrays={k:q[k] for k in q.files};keys=[base64.b64decode(x) for x in meta['development_keys_b64']]
    tr={
      'RAW':{'U':np.zeros_like(arrays['U_PLUS']),'B':None},
      'PAIR20':{'U':arrays['U_PLUS'],'B':None},
      'PAIR-RANDOM-PROTECT8':{'U':arrays['U_PLUS'],'B':arrays['B_RANDOM']},
      'SIGNED-NOGATE':{'U':arrays['U_PLUS'],'B':arrays['B_ALL']},
      'SIGNED-SHUFFLED-NEG':{'U':arrays['U_PLUS'],'B':arrays['B_SHUF']},
      'SIGNED-GATED':{'U':arrays['U_PLUS'],'B':arrays['B_GATE']},
    }
    return meta,tr,keys

def run_development_worker(task):
    seed,out_dir,smoke=task;out_dir=Path(out_dir);d=out_dir/'development';d.mkdir(parents=True,exist_ok=True);meta=d/f'S{seed}.json'
    if meta.exists():
        try:
            md=json.loads(meta.read_text())
            if md.get('development_identity')==development_identity(seed,smoke):return {'seed':seed,'status':'reused'}
        except Exception:pass
    t=time.time();dev=learn_development(seed,smoke);md=save_development(out_dir,dev,smoke)
    return {'seed':seed,'status':'new','positive_pair_updates':md['positive_pair_updates'],'negative_gated_updates':md['negative_gated_updates'],'seconds':time.time()-t}


# -----------------------------------------------------------------------------
# Fixed distributed slow memory — only representation differs
# -----------------------------------------------------------------------------

class SlowSystem:
    def __init__(self,seed,transforms):
        self.seed=int(seed);self.transforms=transforms;rr=np.random.default_rng(seed+1000+3604096)
        self.contacts=np.stack([rr.choice(D,SLOW_CONTACTS,replace=False) for _ in range(SLOW_UNITS)]).astype(np.int16)
        self.proj=rr.normal(size=(SLOW_UNITS,SLOW_CONTACTS));self.proj/=np.linalg.norm(self.proj,axis=1,keepdims=True)
        self.W=np.zeros((len(ARMS),SLOW_UNITS,OUTPUTS),dtype=np.float64);self.arm_index={a:i for i,a in enumerate(ARMS)}
        self.updates=np.zeros(len(ARMS),dtype=np.int64);self._cache=collections.OrderedDict();self.cache_cap=8192
    def clear_transient(self):self._cache.clear()
    def rep(self,arm,c):
        z=raw_rep(c);t=self.transforms[arm];return apply_transform(z,t['U'],t['B'])
    def encodings(self,c):
        c=bytes(c)[-12:]
        if c in self._cache:
            q=self._cache.pop(c);self._cache[c]=q;return q
        Z=np.stack([self.rep(a,c) for a in ARMS]);dr=np.tanh(np.einsum('auc,uc->au',Z[:,self.contacts],self.proj,optimize=True))
        idx=np.argpartition(-np.abs(dr),SLOW_K-1,axis=1)[:,:SLOW_K];idx=np.sort(idx,axis=1);vals=np.take_along_axis(dr,idx,axis=1)
        vals/=np.maximum(np.linalg.norm(vals,axis=1,keepdims=True),1e-12)
        q={a:(idx[i].copy(),vals[i].copy()) for i,a in enumerate(ARMS)};self._cache[c]=q
        if len(self._cache)>self.cache_cap:self._cache.popitem(last=False)
        return q
    def scores_all(self,c,enc=None):
        enc=self.encodings(c) if enc is None else enc;out=np.zeros((len(ARMS),OUTPUTS))
        for i,a in enumerate(ARMS):
            ix,v=enc[a];out[i]=v@self.W[i,ix,:]
        return out
    def scores(self,arm,c,encoding=None):
        ix,v=self.encodings(c)[arm] if encoding is None else encoding;i=self.arm_index[arm];return v@self.W[i,ix,:]
    def write_all(self,c,y,enc=None):
        enc=self.encodings(c) if enc is None else enc
        for i,a in enumerate(ARMS):
            ix,v=enc[a];block=self.W[i,ix,:];s=v@block;vv=float(v@v)
            if vv>1e-15:
                e=-s;e[int(y)]+=1.;self.W[i,ix,:]=block+SLOW_RATE*np.outer(v/vv,e);self.updates[i]+=1
    def representation_bytes(self,arm):
        t=self.transforms[arm];return int(t['U'].nbytes+(0 if t['B'] is None else t['B'].nbytes))
    def arm_state_bytes(self,arm):
        i=self.arm_index[arm];return int(self.W[i].nbytes+self.contacts.nbytes+self.proj.nbytes+self.representation_bytes(arm)+128)
    def state_hash(self):
        h=hashlib.sha256();h.update(self.W.tobytes());h.update(self.contacts.tobytes());h.update(self.proj.tobytes());h.update(self.updates.tobytes())
        for a in ARMS:
            t=self.transforms[a];h.update(t['U'].tobytes());h.update(b'none' if t['B'] is None else t['B'].tobytes())
        return h.hexdigest()

@dataclass
class Agg:
    n:int=0;loss_sum:float=0.;correct:int=0
    def add_prob(self,p,y):self.n+=1;self.loss_sum+=-math.log2(max(float(p[int(y)]),1e-300));self.correct+=int(int(np.argmax(p))==int(y))
    def result(self):return {'n':self.n,'bpb':self.loss_sum/self.n if self.n else None,'accuracy':self.correct/self.n if self.n else None}

def deployed_prob(fast_scores,slow_scores):return .5*psoft(fast_scores)+.5*psoft(slow_scores)

def contexts_for_ids(stream,ids,wrappers,current_targets):
    return [(surface_context(stream['keys'][i],w),int(current_targets[i]),int(i),int(w)) for i in ids for w in wrappers]

def probe_distributed(slows,fast,contexts):
    out={a:Agg() for a in ARMS}
    for c,y,_i,_w in contexts:
        pf=psoft(fast.scores(c,count_ops=False));enc=slows.encodings(c);ss=slows.scores_all(c,enc);pp=.5*pf[None,:]+.5*psoft_batch(ss)
        for i,a in enumerate(ARMS):out[a].add_prob(pp[i],y)
    return {a:g.result() for a,g in out.items()}

def probe_checkpoint(stream,current_targets,cp,slows,fast):
    wi=ord(cp['wave'])-65;count=stream['keys_per_wave'];rows=[];upto=min(len(stream['keys']),(wi+1)*count)
    for cohort in range(wi+1):
        ids=list(range(cohort*count,min((cohort+1)*count,len(stream['keys']))));tr=contexts_for_ids(stream,ids,TRAIN_WRAPPERS,current_targets);he=contexts_for_ids(stream,ids,HELDOUT_WRAPPERS,current_targets)
        qtr=probe_distributed(slows,fast,tr);qhe=probe_distributed(slows,fast,he)
        for a in ARMS:rows.append({'checkpoint':cp['name'],'wave':cp['wave'],'cohort':chr(65+cohort),'arm':a,'trained_accuracy':qtr[a]['accuracy'],'trained_bpb':qtr[a]['bpb'],'heldout_accuracy':qhe[a]['accuracy'],'heldout_bpb':qhe[a]['bpb']})
    ids=list(range(upto));he=contexts_for_ids(stream,ids,HELDOUT_WRAPPERS,current_targets);qhe=probe_distributed(slows,fast,he)
    for a in ARMS:rows.append({'checkpoint':cp['name'],'wave':cp['wave'],'cohort':'ALL','arm':a,'trained_accuracy':None,'trained_bpb':None,'heldout_accuracy':qhe[a]['accuracy'],'heldout_bpb':qhe[a]['bpb']})
    return rows


def geometry(slows,stream):
    ids=list(range(len(stream['keys'])));targets=np.asarray(stream['labels']);out=[]
    # Deterministic comparison pairs shared by all arms.
    same=[(i,i) for i in ids]
    by=collections.defaultdict(list)
    for i,y in enumerate(targets):by[int(y)].append(i)
    sameout=[]
    for members in by.values():
        for a,b in zip(members[::2],members[1::2]):sameout.append((a,b))
    # Different-outcome pair by deterministic cyclic search.
    diffout=[]
    for i in ids:
        for off in range(1,len(ids)):
            j=(i+off)%len(ids)
            if targets[j]!=targets[i]:diffout.append((i,j));break
    for a in ARMS:
        ss=[];ds=[];dd=[]
        for i,_ in same:ss.append(float(slows.rep(a,surface_context(stream['keys'][i],4))@slows.rep(a,surface_context(stream['keys'][i],5))))
        for i,j in sameout:ds.append(float(slows.rep(a,surface_context(stream['keys'][i],4))@slows.rep(a,surface_context(stream['keys'][j],4))))
        for i,j in diffout:dd.append(float(slows.rep(a,surface_context(stream['keys'][i],4))@slows.rep(a,surface_context(stream['keys'][j],4))))
        sm=float(np.mean(ss));dsm=float(np.mean(ds)) if ds else None;ddm=float(np.mean(dd)) if dd else None
        out.append({'representation':a,'same_identity':sm,'different_same_outcome':dsm,'different_different_outcome':ddm,
                    'positive_margin':None if dsm is None else sm-dsm,'conflict_margin':None if ddm is None else sm-ddm})
    return out


def probe_bank(stream,current_targets,wave_index,current_id,seed):
    count=stream['keys_per_wave'];prior=list(range(max(0,wave_index*count)))
    if not prior:return []
    def h(i):return hashlib.sha256(f'v47probe:{seed}:{wave_index}:{i}'.encode()).digest()
    chosen=sorted(prior,key=h)[:min(INTERFERENCE_PROBES,len(prior))]
    out=[]
    for i in chosen:
        w=4 if i%2==0 else 5;out.append((surface_context(stream['keys'][i],w),int(current_targets[i]),int(i),int(w)))
    return out

def _losses(slows,fast,arm,contexts):
    return [-math.log2(max(float(deployed_prob(fast.scores(c,count_ops=False),slows.scores(arm,c))[int(y)]),1e-300)) for c,y,_i,_w in contexts]

def audit_before(slows,fast,contexts):return {a:_losses(slows,fast,a,contexts) for a in ARMS}
def audit_after(slows,fast,contexts,before,event_meta,event_index):
    rows=[];cy=int(event_meta['y']);cid=int(event_meta['key_id'])
    for a in ARMS:
        af=_losses(slows,fast,a,contexts);groups={'all':[],'same_identity_alt':[],'same_outcome_diff_identity':[],'different_outcome_diff_identity':[]}
        for j,(_c,py,pid,_w) in enumerate(contexts):
            d=float(af[j]-before[a][j]);groups['all'].append(d)
            if pid==cid:groups['same_identity_alt'].append(d)
            elif int(py)==cy:groups['same_outcome_diff_identity'].append(d)
            else:groups['different_outcome_diff_identity'].append(d)
        for cat,vals in groups.items():rows.append({'event_index':int(event_index),'arm':a,'category':cat,'n_probes':len(vals),'mean_delta_bpb':float(np.mean(vals)) if vals else None,'wave':event_meta['wave'],'pos':event_meta['pos'],'kind':event_meta['kind'],'key_id':cid,'outcome':cy})
    return rows


# -----------------------------------------------------------------------------
# Persistent job, checkpointing
# -----------------------------------------------------------------------------

@dataclass
class RuntimeState:
    record_index:int;event_index:int;history:bytearray;current_targets:np.ndarray;base_enc:object;base:object;fast:object;slows:SlowSystem;groups:dict
    checkpoint_rows:list=field(default_factory=list);revision_rows:list=field(default_factory=list);interference_rows:list=field(default_factory=list);write_events:list=field(default_factory=list);peak_rss_bytes:int=0

def new_runtime(seed,stream,transforms):
    groups={a:{'whole':Agg(),'target':Agg(),'nontarget':Agg()} for a in ARMS}
    return RuntimeState(0,0,bytearray(),np.array(stream['labels'],dtype=np.int64),V39.Encoder(seed,'last_fd'),V39.Learner('full'),
                        CAND_A.SparseLayer(seed,FAST_UNITS,FAST_K,FAST_CONTACTS,FAST_RATE),SlowSystem(seed,transforms),groups,peak_rss_bytes=current_rss_bytes())

def clear_transient(state):
    if hasattr(state.fast,'_enc_cache'):state.fast._enc_cache={}
    state.slows.clear_transient()

def runtime_hash(state):
    h=hashlib.sha256();h.update(state.slows.state_hash().encode());h.update(state.fast.output.tobytes());h.update(state.base.state_hash().encode());h.update(bytes(state.history));h.update(state.current_targets.tobytes());h.update(str(state.record_index).encode());h.update(str(state.event_index).encode());return h.hexdigest()

def persistent_identity(seed,load,stream_sha,dev_sha,smoke=False):
    return sha_json({'version':VERSION,'source':SCIENTIFIC_SOURCE_ID,'design':DESIGN_SHA256,'seed':int(seed),'load':int(load),'stream':stream_sha,'development':dev_sha,'smoke':bool(smoke),'numpy':np.__version__})

def save_checkpoint(path,identity,state):clear_transient(state);atomic_pickle(path,{'identity':identity,'state':state})
def load_checkpoint(path,identity):
    o=load_pickle(path)
    if o.get('identity')!=identity:raise RuntimeError('checkpoint identity mismatch')
    return o['state']

def base_miss_schedule(seed,load,stream):
    enc=V39.Encoder(seed,'last_fd');base=V39.Learner('full');hist=bytearray();ordinary=set();meta={};idx=0
    for row in stream['rows']:
        for pos,y in enumerate(row['raw']):
            z,b=enc.encode(hist);s=base.scores(z,b)
            if len(hist)>=12 and int(np.argmax(s))!=int(y):ordinary.add(idx)
            meta[idx]={'wave':row['wave'],'pos':int(pos),'kind':row['kind'],'key_id':int(row['key_id']),'y':int(y)}
            base.observe(z,b,int(y));hist.append(int(y));
            if len(hist)>12:del hist[:-12]
            idx+=1
    return ordinary,meta

def sample_events(seed,events,maximum=INTERFERENCE_MAX):
    return set(sorted(events,key=lambda i:hashlib.sha256(f'v47:{seed}:{int(i)}'.encode()).digest())[:min(maximum,len(events))])

def run_persistent_job(seed,load,out_dir,smoke=False,stop_after_records=None):
    out_dir=Path(out_dir);jobs=out_dir/'jobs';jobs.mkdir(parents=True,exist_ok=True);md,tr,devkeys=load_development(out_dir,seed)
    stream=make_persistent_stream(seed,load,exclude_keys=devkeys,repeats=REPEATS,smoke=smoke);assert not (set(stream['keys']) & set(devkeys)),'development/persistent identity overlap';ident=persistent_identity(seed,load,stream['raw_sha256'],md['npz_sha256'],smoke)
    result_path=jobs/f'L{load}_S{seed}.json.gz';checkpoint=jobs/f'L{load}_S{seed}.checkpoint.pkl'
    if result_path.exists():
        try:
            x=load_gzip_json(result_path)
            if x.get('identity')==ident:return {'status':'reused','result':x['result']}
        except Exception:pass
    writes,event_meta=base_miss_schedule(seed,load,stream);sample=sample_events(seed,writes)
    state=load_checkpoint(checkpoint,ident) if checkpoint.exists() else new_runtime(seed,stream,tr)
    cpmap=collections.defaultdict(list)
    for cp in stream['checkpoints']:cpmap[cp['record_count']].append(cp)
    t0=time.time()
    for ri in range(state.record_index,len(stream['rows'])):
        row=stream['rows'][ri]
        if row['kind']=='revision':state.current_targets[int(row['key_id'])]=int(row['target'])
        wave_index=max(0,ord(row['wave'])-65)
        for pos,y in enumerate(row['raw']):
            idx=state.event_index;z,b=state.base_enc.encode(state.history);sb=state.base.scores(z,b);fs=state.fast.scores(state.history);pf=psoft(fs)
            if len(state.history)>=12:
                c=bytes(state.history)[-12:];encs=state.slows.encodings(c);ss=state.slows.scores_all(c,encs);pp=.5*pf[None,:]+.5*psoft_batch(ss)
            else:
                c=None;encs=None;pp=np.broadcast_to(.5*pf[None,:]+.5*np.full((1,OUTPUTS),1./OUTPUTS),(len(ARMS),OUTPUTS))
            for ai,a in enumerate(ARMS):state.groups[a]['whole'].add_prob(pp[ai],y);state.groups[a]['target' if pos==12 else 'nontarget'].add_prob(pp[ai],y)
            before=None;contexts=[]
            if idx in writes and c is not None:
                if idx in sample:
                    contexts=probe_bank(stream,state.current_targets,wave_index,int(row['key_id']),seed)
                    if contexts:before=audit_before(state.slows,state.fast,contexts)
                state.slows.write_all(c,int(y),encs);state.write_events.append(int(idx))
                if before is not None:state.interference_rows.extend(audit_after(state.slows,state.fast,contexts,before,event_meta[idx],idx))
            state.base.observe(z,b,int(y));state.fast.write(state.history,int(y));state.history.append(int(y))
            if len(state.history)>12:del state.history[:-12]
            state.event_index+=1
        state.record_index=ri+1;state.peak_rss_bytes=max(state.peak_rss_bytes,current_rss_bytes())
        for cp in cpmap.get(state.record_index,[]):
            rr=probe_checkpoint(stream,state.current_targets,cp,state.slows,state.fast);state.checkpoint_rows.extend(rr)
            if cp['type']=='revision':state.revision_rows.extend([dict(r,revision_exposure=cp['exposure']) for r in rr if r['cohort']=='A'])
        if state.record_index%128==0 or cpmap.get(state.record_index):save_checkpoint(checkpoint,ident,state)
        if stop_after_records is not None and state.record_index>=stop_after_records:
            save_checkpoint(checkpoint,ident,state);return {'status':'partial','record_index':state.record_index,'identity':ident}
    final=[r for r in state.checkpoint_rows if r['checkpoint']=='after_D'];terminal=[]
    for a in ARMS:
        ar=next(r for r in final if r['arm']==a and r['cohort']=='ALL');dr=next(r for r in final if r['arm']==a and r['cohort']=='D');g={k:v.result() for k,v in state.groups[a].items()}
        terminal.append({'seed':seed,'load':load,'arm':a,'whole_bpb':g['whole']['bpb'],'whole_accuracy':g['whole']['accuracy'],'target_bpb':g['target']['bpb'],'target_accuracy':g['target']['accuracy'],'nontarget_bpb':g['nontarget']['bpb'],'D_after_D_trained_accuracy':dr['trained_accuracy'],'D_after_D_trained_bpb':dr['trained_bpb'],'heldout_after_D_accuracy':ar['heldout_accuracy'],'heldout_after_D_bpb':ar['heldout_bpb']})
    geo=geometry(state.slows,stream);costs=[]
    for a in ARMS:costs.append({'seed':seed,'load':load,'arm':a,'slow_semantic_bytes':state.slows.arm_state_bytes(a),'slow_matrix_bytes':int(state.slows.W[state.slows.arm_index[a]].nbytes),'projection_bytes':int(state.slows.contacts.nbytes+state.slows.proj.nbytes),'representation_bytes':state.slows.representation_bytes(a),'writes':int(state.slows.updates[state.slows.arm_index[a]])})
    by=collections.defaultdict(list)
    for x in state.interference_rows:
        if x['mean_delta_bpb'] is not None:by[(x['arm'],x['category'])].append(x['mean_delta_bpb'])
    isum=[{'seed':seed,'load':load,'arm':a,'category':cat,'n_events':len(v),'mean_delta_bpb':float(np.mean(v))} for (a,cat),v in by.items()]
    whash=sha_json(state.write_events)
    result={'seed':seed,'load':load,'identity':ident,'stream_sha256':stream['raw_sha256'],'development_sha256':md['npz_sha256'],'terminal':terminal,'checkpoints':state.checkpoint_rows,'revision':state.revision_rows,'geometry':geo,'state_costs':costs,'interference_events':state.interference_rows,'interference_summary':isum,'write_schedule':{'count':len(writes),'hash':sha_json(sorted(writes)),'observed_count':len(state.write_events),'observed_hash':whash},'sample_budget':{'development_raw_bytes':md['development_raw_bytes'],'persistent_raw_bytes':len(stream['raw']),'persistent_records':len(stream['rows'])},'final_state_hash':runtime_hash(state),'peak_rss_bytes':state.peak_rss_bytes,'elapsed_seconds':time.time()-t0}
    atomic_gzip_json(result_path,{'identity':ident,'result':result});
    if checkpoint.exists():checkpoint.unlink()
    return {'status':'new','result':result}

def run_persistent_worker(task):seed,load,out_dir,smoke=task;return run_persistent_job(int(seed),int(load),Path(out_dir),bool(smoke))


# -----------------------------------------------------------------------------
# Self-test and engineering firewall
# -----------------------------------------------------------------------------

def static_self_test():
    checks={};seed=47991;p=make_ids(seed,32);U,audit,false=learn_positive(seed,p,True);checks['positive_rank']=int(np.linalg.matrix_rank(U));assert checks['positive_rank']==20
    n=make_ids(seed+1,24,exclude=p.keys);neg=learn_negative(seed,U,n,True);tr=build_transforms(seed,U,neg);assert set(tr)==set(ARMS);checks['negative_gated_updates']=neg['gated_updates']
    # Every representation must stay unit norm and preserve dimensions.
    c=surface_context(n.keys[0],4)
    for a,t in tr.items():
        z=apply_transform(raw_rep(c),t['U'],t['B']);assert z.shape==(D,);assert abs(np.linalg.norm(z)-1)<1e-8
    slow=SlowSystem(seed,tr);enc=slow.encodings(c);assert all(len(enc[a][0])==SLOW_K for a in ARMS)
    before=slow.state_hash();slow.scores_all(c,enc);assert slow.state_hash()==before;slow.write_all(c,65,enc);assert np.all(slow.updates==1)
    checks['slow_active_units']=SLOW_K;checks['common_write_updates']=slow.updates.tolist()
    # Short predictor is causal: prediction before update differs after observation.
    sp=ShortPredictor();p0=sp.probability(1,2);sp.observe(1,2);p1=sp.probability(1,2);assert p1>p0;checks['short_causal']=True
    # Persistent state is bounded and contains no object slots/pathways.
    assert max(slow.arm_state_bytes(a) for a in ARMS)<=STATE_CAP_BYTES;checks['max_state_bytes']=max(slow.arm_state_bytes(a) for a in ARMS)
    checks['no_slots']=not hasattr(slow,'slots');checks['passed']=True;return checks

def engineering_check(out_dir,workers=2):
    out_dir=Path(out_dir);out_dir.mkdir(parents=True,exist_ok=True);seeds=ENGINEERING_SEEDS
    # Development smoke.
    for s in seeds:run_development_worker((s,out_dir,True))
    # Forced partial + resume equivalence on first engineering seed.
    s=seeds[0];load=64
    part=run_persistent_job(s,load,out_dir,True,stop_after_records=5);assert part['status']=='partial'
    resumed=run_persistent_job(s,load,out_dir,True);rh=resumed['result']['final_state_hash']
    # Fresh copy in temp directory.
    with tempfile.TemporaryDirectory() as td:
        td=Path(td);run_development_worker((s,td,True));fresh=run_persistent_job(s,load,td,True);fh=fresh['result']['final_state_hash']
    resume_ok=(rh==fh)
    atomic_json(out_dir/'RESUME_AUDIT.json',{'passed':resume_ok,'resumed_hash':rh,'fresh_hash':fh})
    # Multiprocessing smoke, disjoint seed.
    tasks=[(seeds[1],64,str(out_dir),True)];run_parallel(tasks,run_persistent_worker,max(1,min(2,int(workers))))
    # Firewall checks from resulting jobs.
    rec=[]
    for p in (out_dir/'jobs').glob('*.json.gz'):
        try:rec.append(load_gzip_json(p)['result'])
        except Exception:pass
    ok=bool(rec) and all(r['write_schedule']['count']==r['write_schedule']['observed_count'] and r['write_schedule']['hash']==r['write_schedule']['observed_hash'] for r in rec)
    atomic_json(out_dir/'FIREWALL_AUDIT.json',{'passed':bool(ok and resume_ok),'resume':resume_ok,'jobs_checked':len(rec),'no_slots':True,'common_write_schedule':ok})
    return {'passed':bool(ok and resume_ok),'jobs':len(rec)}


# -----------------------------------------------------------------------------
# Aggregation and gates
# -----------------------------------------------------------------------------

def collect_results(out_dir):
    out=[]
    for p in sorted((Path(out_dir)/'jobs').glob('*.json.gz')):
        try:out.append(load_gzip_json(p)['result'])
        except Exception:pass
    return out

def write_csv(path,rows,gz=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return path.write_text('')
    fields=sorted({k for r in rows for k in r})
    fh=gzip.open(path,'wt',newline='') if gz else path.open('w',newline='')
    with fh:
        w=csv.DictWriter(fh,fieldnames=fields);w.writeheader();w.writerows([{k:r.get(k) for k in fields} for r in rows])

def paired(terminal,left,right,key,load=PRIMARY_LOAD):
    by=collections.defaultdict(dict)
    for r in terminal:
        if r['load']==load:by[int(r['seed'])][r['arm']]=r
    vals=[]
    for d in by.values():
        if left in d and right in d and d[left].get(key) is not None and d[right].get(key) is not None:vals.append(float(d[left][key])-float(d[right][key]))
    return interval(vals)

def aggregate(out_dir,science_seeds=SCIENCE_SEEDS):
    out_dir=Path(out_dir);results=[r for r in collect_results(out_dir) if int(r['seed']) in set(science_seeds) and int(r['load']) in (PRIMARY_LOAD,SECONDARY_LOAD)]
    terminal=[x for r in results for x in r['terminal']];geo=[];costs=[];isum=[];ie=[];rev=[];checks=[];budget=[]
    for r in results:
        geo.extend([dict(x,seed=r['seed'],load=r['load']) for x in r['geometry']]);costs.extend(r['state_costs']);isum.extend(r['interference_summary']);ie.extend([dict(x,seed=r['seed'],load=r['load']) for x in r['interference_events']]);rev.extend([dict(x,seed=r['seed'],load=r['load']) for x in r['revision']]);checks.extend([dict(x,seed=r['seed'],load=r['load']) for x in r['checkpoints']]);budget.append(dict(seed=r['seed'],load=r['load'],**r['sample_budget']))
    # Development metadata.
    devrows=[];negrows=[];shortrows=[];shuffle=[]
    for s in science_seeds:
        p=out_dir/'development'/f'S{s}.json'
        if not p.exists():continue
        md=json.loads(p.read_text())
        devrows.append({'seed':s,'positive_pair_updates':md['positive_pair_updates'],'positive_false_pair_rate':md['positive_false_pair_rate'],'positive_rank':md['positive_rank'],'positive_orth_error':md['positive_orth_error']})
        eg=_float_vector(md.get('negative_eig_gate'))
        negrows.append({'seed':s,'all_conflicts':md['negative_all_conflicts'],'target_conflicts':md['negative_target_conflicts'],'gated_updates':md['negative_gated_updates'],'verifier_peak':md['negative_verifier_peak'],'protected_rank':PROTECT_RANK,'min_selected_eigen':float(min(eg)) if eg else None})
        shortrows.append({'seed':s,'short_context_bytes':md['short_context_bytes'],'surprise_threshold':SHORT_SURPRISE})
        row={'seed':s,'gated_updates':md['negative_gated_updates'],'source_surprise_mae':md.get('shuffle_source_surprise_mae'),'original_similarity_mean':md.get('shuffle_original_similarity_mean'),'shuffled_similarity_mean':md.get('shuffle_similarity_mean'),'unique_source_fraction':md.get('shuffle_unique_source_fraction'),'mean_block_distance':md.get('shuffle_mean_block_distance')};row.update({f'shuffle_{k}':v for k,v in (md.get('shuffle_levels') or {}).items()});shuffle.append(row)
    # Geometry contrasts at primary load.
    gm=collections.defaultdict(dict)
    for x in geo:
        if x['load']==PRIMARY_LOAD:gm[x['seed']][x['representation']]=x
    def gpair(left,right,key):return interval([d[left][key]-d[right][key] for d in gm.values() if left in d and right in d and d[left].get(key) is not None and d[right].get(key) is not None])
    same_delta=gpair(PRIMARY_ARM,PAIR_ARM,'same_identity');diff_delta=gpair(PRIMARY_ARM,PAIR_ARM,'different_different_outcome');margin_delta=gpair(PRIMARY_ARM,PAIR_ARM,'conflict_margin')
    # Interference summaries per seed.
    ib=collections.defaultdict(lambda:collections.defaultdict(dict))
    for x in isum:
        if x['load']==PRIMARY_LOAD:ib[x['seed']][x['arm']][x['category']]=x['mean_delta_bpb']
    def ipair(left,right,cat):return interval([d[left][cat]-d[right][cat] for d in ib.values() if left in d and right in d and cat in d[left] and cat in d[right]])
    idiff=ipair(PRIMARY_ARM,PAIR_ARM,'different_outcome_diff_identity');tsame=ipair(PRIMARY_ARM,PAIR_ARM,'same_identity_alt')
    # Persistent contrasts.
    persist=paired(terminal,PRIMARY_ARM,PAIR_ARM,'heldout_after_D_bpb');acq=paired(terminal,PRIMARY_ARM,PAIR_ARM,'D_after_D_trained_accuracy');whole=paired(terminal,PRIMARY_ARM,PAIR_ARM,'whole_bpb')
    randctl=paired(terminal,'PAIR-RANDOM-PROTECT8',PRIMARY_ARM,'heldout_after_D_bpb') # positive => primary better
    shufctl=paired(terminal,'SIGNED-SHUFFLED-NEG',PRIMARY_ARM,'heldout_after_D_bpb')
    nogate=paired(terminal,'SIGNED-NOGATE',PRIMARY_ARM,'heldout_after_D_bpb')
    # Revision-8 comparison.
    rby=collections.defaultdict(dict)
    for x in rev:
        if x['load']==PRIMARY_LOAD and x.get('revision_exposure')==8:rby[x['seed']][x['arm']]=x
    revacc=interval([d[PRIMARY_ARM]['trained_accuracy']-d[PAIR_ARM]['trained_accuracy'] for d in rby.values() if PRIMARY_ARM in d and PAIR_ARM in d]);revbpb=interval([d[PRIMARY_ARM]['trained_bpb']-d[PAIR_ARM]['trained_bpb'] for d in rby.values() if PRIMARY_ARM in d and PAIR_ARM in d])
    # Scaling gains (PAIR - SIGNED, positive is good).
    g128=paired(terminal,PAIR_ARM,PRIMARY_ARM,'heldout_after_D_bpb',SECONDARY_LOAD);g256=paired(terminal,PAIR_ARM,PRIMARY_ARM,'heldout_after_D_bpb',PRIMARY_LOAD)
    # Gates.
    fw=out_dir/'FIREWALL_AUDIT.json';ra=out_dir/'RESUME_AUDIT.json';n_primary=sum(1 for r in results if r['load']==PRIMARY_LOAD);n_secondary=sum(1 for r in results if r['load']==SECONDARY_LOAD)
    science_common_writes=all(len({x['writes'] for x in r['state_costs'] if x['arm'] in ARMS})==1 and r['write_schedule']['count']==r['write_schedule']['observed_count'] and r['write_schedule']['hash']==r['write_schedule']['observed_hash'] for r in results)
    g0=bool(fw.exists() and ra.exists() and json.loads(fw.read_text()).get('passed') and json.loads(ra.read_text()).get('passed') and n_primary==16 and science_common_writes)
    pstats=interval([r['positive_pair_updates'] for r in devrows]);fstats=interval([r['positive_false_pair_rate'] for r in devrows]);pair_vs_raw=gpair('PAIR20','RAW','positive_margin')
    g1=bool(len(devrows)==16 and pstats['mean'] is not None and pstats['mean']>=300 and fstats['mean'] is not None and fstats['mean']<=.05 and all(r['positive_rank']==20 for r in devrows) and pair_vs_raw['mean'] is not None and pair_vs_raw['mean']>=.05 and pair_vs_raw['low'] is not None and pair_vs_raw['low']>0)
    nstats=interval([r['gated_updates'] for r in negrows]);g2=bool(len(negrows)==16 and nstats['mean'] is not None and nstats['mean']>=100 and min(r['gated_updates'] for r in negrows)>=50 and all((r['min_selected_eigen'] or 0)>1e-12 for r in negrows))
    def levels(arm,key):return [r[key] for r in terminal if r['load']==PRIMARY_LOAD and r['arm']==arm]
    pa=levels(PAIR_ARM,'D_after_D_trained_accuracy');sa=levels(PRIMARY_ARM,'D_after_D_trained_accuracy');g3=bool(len(pa)==16 and len(sa)==16 and np.mean(pa)>=.90 and np.mean(sa)>=.90 and min(pa)>=.80 and min(sa)>=.80)
    g4=bool(same_delta['mean'] is not None and same_delta['mean']>=-.05 and same_delta['low'] is not None and same_delta['low']>=-.08 and diff_delta['mean'] is not None and diff_delta['mean']<=-.05 and diff_delta['high'] is not None and diff_delta['high']<0 and margin_delta['mean'] is not None and margin_delta['mean']>=.05 and margin_delta['low'] is not None and margin_delta['low']>0)
    g5=bool(idiff['mean'] is not None and idiff['mean']<=-.00075 and idiff['high'] is not None and idiff['high']<0 and tsame['mean'] is not None and tsame['mean']<=.05 and tsame['high'] is not None and tsame['high']<=.08)
    g6=bool(persist['mean'] is not None and persist['mean']<=-.10 and persist['high'] is not None and persist['high']<0)
    g7rand=bool(randctl['mean'] is not None and randctl['mean']>=.05 and randctl['low'] is not None and randctl['low']>0);g7shuf=bool(shufctl['mean'] is not None and shufctl['mean']>=.05 and shufctl['low'] is not None and shufctl['low']>0);g7=bool(g7rand and g7shuf)
    # Short gate supported if NOGATE - GATED >= .03 and harmful interference lower for gated.
    inog=ipair('SIGNED-NOGATE',PRIMARY_ARM,'different_outcome_diff_identity');g8=bool(nogate['mean'] is not None and nogate['mean']>=.03 and nogate['low'] is not None and nogate['low']>0 and inog['mean'] is not None and inog['mean']>0)
    g9=bool(whole['mean'] is not None and whole['mean']<=.02 and whole['high'] is not None and whole['high']<=.04 and revacc['mean'] is not None and revacc['mean']>=-.05 and revbpb['high'] is not None and revbpb['high']<=.10)
    scale_complete=(n_secondary==16)
    g10=(bool(g6 and g256['mean'] is not None and g128['mean'] is not None and g256['mean']>=.75*max(g128['mean'],.05)) if scale_complete else None)
    dist=[x for x in costs if x['arm'] in ARMS];g11=bool(dist and max(x['slow_semantic_bytes'] for x in dist)<=STATE_CAP_BYTES)
    for a in ARMS:
        lv={x['load']:x['slow_semantic_bytes'] for x in dist if x['arm']==a}
        if PRIMARY_LOAD in lv and SECONDARY_LOAD in lv and lv[PRIMARY_LOAD]!=lv[SECONDARY_LOAD]:g11=False
    if not g0:verdict='IMPLEMENTATION_INVALID'
    elif not g1:verdict='POSITIVE_DEVELOPMENT_INVALID'
    elif not g2:verdict='NEGATIVE_DEVELOPMENT_INVALID'
    elif not g3:verdict='INVALID_ACQUISITION'
    elif not g4:verdict='NO_SIGNED_GEOMETRY_GAIN'
    elif not g5:verdict='NO_SELECTIVE_INTERFERENCE_GAIN'
    elif not g6:verdict='INTERFERENCE_REDUCED_NO_PERSISTENT_GAIN'
    elif not g7:
        verdict='RANK_REDUCTION_SUFFICIENT' if not g7rand else 'NEGATIVE_CONTENT_SPECIFICITY_FAILED'
    elif not g9:verdict='PERSISTENCE_GAIN_WITH_SAFETY_COST'
    elif not g11:verdict='STATE_BUDGET_FAIL'
    elif g10 is None:verdict='PRIMARY_CLEARED_SCALE_PENDING'
    elif not g10:verdict='GAIN_COLLAPSES_WITH_LOAD'
    else:verdict='SIGNED_REPRESENTATION_QUALIFIED' if g8 else 'SIGNED_LEARNING_SUPPORTED_SHORT_GATE_UNNECESSARY'
    gates={'G0_implementation':g0,'G1_positive_development':g1,'G1_pair_vs_raw':pair_vs_raw,'G2_negative_development':g2,'G2_gated_updates':nstats,'G3_acquisition':g3,'G4_signed_geometry':g4,'G4_same_identity_delta':same_delta,'G4_diff_outcome_delta':diff_delta,'G4_conflict_margin_delta':margin_delta,'G5_selective_interference':g5,'G5_harmful_delta':idiff,'G5_same_identity_transfer_delta':tsame,'G6_persistent_gain':g6,'G6_signed_minus_pair_heldout_bpb':persist,'G7_specificity':g7,'G7_random_minus_signed':randctl,'G7_shuffled_minus_signed':shufctl,'G8_short_filter':g8,'G8_nogate_minus_gated':nogate,'G8_nogate_harmful_minus_gated':inog,'G9_safety':g9,'G9_whole_bpb_delta':whole,'G9_revision8_accuracy_delta':revacc,'G9_revision8_bpb_delta':revbpb,'G10_scaling':g10,'G10_scale_complete':scale_complete,'G10_gain128':g128,'G10_gain256':g256,'G11_bounded_state':g11,'science_common_writes':science_common_writes,'primary_jobs':n_primary,'secondary_jobs':n_secondary,'verdict':verdict}
    # Outputs.
    write_csv(out_dir/'DEVELOPMENT_POSITIVE.csv',devrows);write_csv(out_dir/'DEVELOPMENT_NEGATIVE.csv',negrows);write_csv(out_dir/'SHORT_CONTEXT_AUDIT.csv',shortrows);write_csv(out_dir/'NEGATIVE_SHUFFLE_AUDIT.csv',shuffle);write_csv(out_dir/'REPRESENTATION_GEOMETRY.csv',geo);write_csv(out_dir/'INTERFERENCE_SUMMARY.csv',isum);write_csv(out_dir/'INTERFERENCE_EVENTS.csv.gz',ie,gz=True);write_csv(out_dir/'TERMINAL.csv',terminal);write_csv(out_dir/'CHECKPOINT_METRICS.csv.gz',checks,gz=True);write_csv(out_dir/'REVISION.csv',rev);write_csv(out_dir/'STATE_COSTS.csv',costs);write_csv(out_dir/'SAMPLE_BUDGET.csv',budget)
    contrasts=[]
    for name,st in [('SIGNED-PAIR',persist),('RANDOMPROTECT-SIGNED',randctl),('SHUFFLED-SIGNED',shufctl),('NOGATE-SIGNED',nogate)]:
        row={'contrast':name};row.update({f'heldout_bpb_{k}':v for k,v in st.items()});contrasts.append(row)
    write_csv(out_dir/'PERSISTENCE_CONTRASTS.csv',contrasts);atomic_json(out_dir/'GATES.json',gates)
    manifest={'version':VERSION,'source_sha256':source_sha(),'scientific_runner_sha256':SCIENTIFIC_SOURCE_ID,'reporting_hotfix':'legacy NumPy-array JSON parser + JSON-safe serialization','design_sha256':DESIGN_SHA256,'science_seeds':list(science_seeds),'loads':[PRIMARY_LOAD,SECONDARY_LOAD],'numpy':np.__version__,'scipy':None if scipy is None else scipy.__version__,'platform':platform.platform(),'records':len(results),'verdict':verdict,'fixed_constants':{'pos_rank':POS_RANK,'protect_rank':PROTECT_RANK,'neg_tau':NEG_TAU,'surprise':SHORT_SURPRISE,'slow_units':SLOW_UNITS,'slow_k':SLOW_K,'slow_rate':SLOW_RATE}}
    atomic_json(out_dir/'MANIFEST.json',manifest)
    design_path=SOURCE.with_name('V47_signed_predictive_representation_design.md')
    if design_path.exists():(out_dir/'PROTOCOL.md').write_text(design_path.read_text())
    else:(out_dir/'PROTOCOL.md').write_text(f'# V47 locked protocol\n\nDesign SHA256 `{DESIGN_SHA256}`\n')
    lines=['# V47 results — Signed Predictive Representation Learning','',f'**Verdict:** `{verdict}`','',*[f'- {k}: **{v}**' for k,v in gates.items() if k.startswith('G') and isinstance(v,bool)],'',f"SIGNED-GATED − PAIR20 held-out bpb: {persist['mean']} [{persist['low']}, {persist['high']}]",f"Harmful interference delta: {idiff['mean']} [{idiff['low']}, {idiff['high']}]",f"Conflict-margin delta: {margin_delta['mean']} [{margin_delta['low']}, {margin_delta['high']}]",'']
    (out_dir/'SUMMARY.md').write_text('\n'.join(lines))
    return gates


def package_upload(out_dir):
    out_dir=Path(out_dir);target=out_dir/'v47_results_upload.zip';include=['MANIFEST.json','PROTOCOL.md','GATES.json','SUMMARY.md','DEVELOPMENT_POSITIVE.csv','DEVELOPMENT_NEGATIVE.csv','SHORT_CONTEXT_AUDIT.csv','NEGATIVE_SHUFFLE_AUDIT.csv','REPRESENTATION_GEOMETRY.csv','INTERFERENCE_SUMMARY.csv','INTERFERENCE_EVENTS.csv.gz','PERSISTENCE_CONTRASTS.csv','REVISION.csv','TERMINAL.csv','CHECKPOINT_METRICS.csv.gz','STATE_COSTS.csv','SAMPLE_BUDGET.csv','RESUME_AUDIT.json','FIREWALL_AUDIT.json']
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name in include:
            p=out_dir/name
            if p.exists():z.write(p,arcname=name)
        z.write(SOURCE,arcname=SOURCE.name)
    return target


def run_parallel(tasks,worker,workers):
    if not tasks:return []
    if int(workers)<=1:return [worker(t) for t in tasks]
    ctx=mp.get_context('spawn');out=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=int(workers),mp_context=ctx) as ex:
        futs=[ex.submit(worker,t) for t in tasks]
        for f in concurrent.futures.as_completed(futs):out.append(f.result())
    return out


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--self-test',action='store_true');ap.add_argument('--engineering-check',action='store_true')
    ap.add_argument('--stage',choices=('development','primary','scale','all','aggregate'),default='all');ap.add_argument('--out-dir',type=Path,default=Path('v47_results'));ap.add_argument('--workers',type=int,default=int(os.environ.get('V47_WORKERS','4')));ap.add_argument('--seeds',type=int,help='Non-scientific prefix of science roster.');ap.add_argument('--smoke',action='store_true');ap.add_argument('--skip-engineering',action='store_true')
    args=ap.parse_args(argv)
    if args.self_test:
        print(json.dumps(static_self_test(),indent=2));return 0
    out=args.out_dir;out.mkdir(parents=True,exist_ok=True)
    if args.engineering_check:
        print(json.dumps(engineering_check(out,args.workers),indent=2));return 0
    seeds=list(SCIENCE_SEEDS[:args.seeds] if args.seeds else SCIENCE_SEEDS)
    scientific=not args.smoke and args.seeds is None
    if scientific and not args.skip_engineering and args.stage not in ('aggregate',):
        if not (out/'FIREWALL_AUDIT.json').exists() or not json.loads((out/'FIREWALL_AUDIT.json').read_text()).get('passed'):
            engineering_check(out,args.workers)
    if args.stage in ('development','primary','scale','all'):
        devtasks=[(s,str(out),args.smoke) for s in seeds];run_parallel(devtasks,run_development_worker,args.workers)
    if args.stage in ('primary','all'):
        run_parallel([(s,PRIMARY_LOAD,str(out),args.smoke) for s in seeds],run_persistent_worker,args.workers)
    if args.stage in ('scale','all'):
        run_parallel([(s,SECONDARY_LOAD,str(out),args.smoke) for s in seeds],run_persistent_worker,args.workers)
    if args.stage=='development':
        print(json.dumps({'status':'development_complete','out_dir':str(out),'seeds':len(seeds)},indent=2));return 0
    gates=aggregate(out,seeds if args.seeds is not None else SCIENCE_SEEDS)
    target=package_upload(out)
    print(json.dumps({'verdict':gates.get('verdict'),'upload':str(target),'out_dir':str(out)},indent=2));return 0

if __name__=='__main__':raise SystemExit(main())
