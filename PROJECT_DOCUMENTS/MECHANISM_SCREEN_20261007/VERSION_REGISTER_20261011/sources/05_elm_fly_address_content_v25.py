#!/usr/bin/env python3
"""ELM V25: stable addresses, plastic content and local bank learning.

ONE COMMAND (Python 3.10+, CPU, including Mac M3):
    python3 elm_fly_address_content_v25.py --install-deps

Self-contained: embedded fly mappings, NumPy and PyTorch. No data files or
pretrained checkpoints required. Every active candidate develops without hidden
backpropagation. Prescribed microdynamics and population normalization remain
explicit modeling assumptions. Inactive V24 historical code is retained solely
for exact arithmetic continuity and is never scheduled by the V25 manifest.

Default: 58 configurations x six fresh seeds = 348 trajectories. Address/value
stability, write rate/release, practical stable halves and local novelty gating,
minimal temporal encoding, stored int8 readouts, structured/context controls,
delayed teaching and matched-budget reward exploration are predeclared.
Hard/blended reads share learned state and do not multiply training exposures.

macOS caffeinate starts automatically and exits with this process. It prevents
idle system sleep; it does not override closing a laptop lid or forced sleep.
Use --no-caffeinate to disable it. Other platforms run normally.

One compact *_results.zip is refreshed during the run. Completed jobs resume
individually with --resume FOLDER; original experiment arguments are restored.
Sensory-only feature caches remain in that folder and are excluded from the ZIP.
--list-jobs prints the roster without installing packages.
--self-test validates arithmetic and scientific interfaces.
--smoke runs every configuration on reduced data (execution check only).
--max-jobs N stops after N new jobs while preserving a resumable partial archive.
"""
from __future__ import annotations
import argparse
import base64
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import zipfile

VERSION = '23.0.0'
ARMS = ('uniform_delta', 'g1_feedback', 'joint_feedback', 'joint_scrambled',
        'joint_outcome_only', 'cumulative_ridge')
SOURCE_V21_SHA256 = '1b8d47ddb21f654fc7df9ba879c7c316e5b6e1a31b0d1c6e9ebb8240989931b8'

# Embedded measured anatomy for V22; assembled into the one-file runner.
ANATOMY_SHA256 = '66d591704fc89ddc46c02bc24fd60615724daa483cc0173abc0d67dae260a044'
ANATOMY_B85 = '''P)h>@6aWAK2mk;8Apq3Ec_ea~0|3C%0{{#F002f`b7Ns_Wn?aHaCw}Y<yTc-y!AoEAOxjUP(r$q&Pg{C(%s!H-Q6i5A{~N3NC_eaDyV>>5~6}Aq9O(o5|=y1
bANxpv)-Nk;+(zqnsa_XbBulR&l&0HnGq5M6NGa(It4iRb4W39$T&-I@G@~Y`}ha?+j-mg_&YlN?{!r>&j6>r>jAEIzD|3eIfVp8c$tI+g}8W`BANcb7754U
8&`%FYp_B}Shu?sgwLCw0`GbTBJS?Lq}|k9R7;TT*J`K&Pru!>uMCMGS&y<l?N)|YBtCc2a+0ycZqH9iWQV&T-Tcg&0**;x^4nzb$oiHTNR|-;4PGTd;zDm!
d7oQ)+vkSIgggr`H9K)#h08%B^A^Sm6Loh30+8Xszwak&It*Og%vBz$VDrxe_r*_Pkl&zj@-0aLwTxdV_eK&Xg2#HA7b-#1`BPMNEeL;B^V1)0B!GVEfrR@?
8Z_GJ8;#BsKy~%6vSh{uG;lOoyGymec7Jo-tF1CbOX|kF6>k9bP>Y&rX#&KF*j3I&gra@0{F<q=2?`8S(hL%Y(6x5w*#Wy`NH{W=@2)4nmdnYR=2$;;?oK(q
%ZUfmnk$3o%_>m*yX>AtnU4vhU+Z>{djR?ij7(HX*jC$lp6>02hy%oV?Dw<KB4*1yLY@nW?Dtk7*F#ZeH}@*yk3E_>eZuCKi_sXOrz0ZSfU{Gggz8zbpj0~=
CwaXYgcUc6I)9hJurHr*=x_nP5cQ?gSXH97{i6WAb{4jT-qjPz=VLKDCEIqk63=4Jl^!Qeg(V&1@+-|e6eW2pT0P5ynQWhJVz)cK2o7^rn&yJQBIqvla0&ig
Vl8nz;17+9tfMDSB;Z=u^>*RDa<FLl*01SY#P`>>H+F=pu}%N&oJ*oME}qb;s)=jDQLB**vaD*9T-59%pud8YXFF1@=ksvl(B?0r$9=fzze@C6AOs`Kbu$9L
st}h^!)a@u271*I$5g)t40G!V?kgU~=+(2ejHjy*o^d+2?Qkrv-a1-uAd~=#%U?SV)dqu~`*&RWr5p&dX1rJIOu^rKnH#8&$Cu;z{g0VaQDie<855O`I@7%8
J6AFhAwP15WBnrX3hx^H3b}}Xit!>vcB!CaY>u%yQ3K*}JwZQ%4k)^Gu?AMgpsDWF0}H+wRQ8E1avm<iWalD3ds#VZ9_Drp6jy-uapt(*b`AWUzAjQ;FGNKC
*V668HuTKun{X9Jqc7h{u5d65T1Lu@Lwrdf_<w8t8d-FiMmT<a{n4U?GKew<^)~R`#vyvXi*Ym8;PWf(rt-ckDCMGGRFvt&DTP#*gGvF2uHdK942gqmzq9mF
>I~5CD0mc0)`RFIXW%`CIQWpTJGXdOqS7P%1zlV{K08i+j^C~VH+3~@#dZN0<Zo71dnCi+O4eP9lnnU1l#vtt77yxlZicA8j_^&6;Bpa+1FOW<(=WSXL3-76
yfgL!F8(fQ7Jli0*jvLRuJ`?LN3u;kTgn)%)a+EAk)EI(E2$~dw1pLUNC?Yv1ZK21t>PP^p?N*6&Vn@%zsM7BT^6)OK6SMXb5|r(cL<b4d;>t*9rWlabpVVK
+?xls!Vxf@`TLi(1Lmk-AJBOagmo#xNTVO;K=nz@D5};FwVzj{q2h^R#>LrQVMnmow&_36iNQXRUA-e&ez2JTwExmybMUwLly*ONN431-8n3A<R+d^~?L%_F
$lD$CN!S**hn5C4j$c4;c56;afh&aeAD=1a48pbFyzN{!f+5oFrh7v-71N4JiJm(a;2cQ%7AYTxjQO@iO}Aj!e|LNGvBnuqZ04Q}7u|5ybzau|rW8WrNen1f
+;A|w$|#yW9a$O%gq_BcI5Vu#*1q2lM4EqJwdfh+-|Vw9VX_zD#I_-=u+J6(F70jqro8aj_23Je1xu*jX#M-W*$*_nw|{)v(7^Nw)5VOnK>R48$tYt<hH{g}
p-+PDxJ~@o!c8+0$GkXs`R-l7;)u`via}Le;^UF&4fKYQ{SL{zbt*b8S~oPBgyE9J=tu5pJ5<MhV<AqvfM~C?qqcmZcw#ZV+Ux8DL0v&Ux)>Xjo~T}tn-0YA
13}7oQx{A)`Buqs1Vgzz_;id~3|7OpCHYR+g7YH5wN8Fd@NmB<S5fl-C9$IkQ%4S_Jq2|5Y88>2$kUUb?~dsKQ@`9!Z_s_nPuewzfm}((=$NAc%xP{{u*oLi
GT8w?l2=h^Tv~c^WWp8CmZ_Tdho?h0_7Y#pH7nFNJveg=*4TfkyGgUv7q_mSWnk^w%lDVD3n7YjC>x-?B>gB9iMX?0Sk?{`|EBIUobrS2BhTC6dD*aUKlAz!
qXuq#tV+Ew8x8)nFODPLR!EDW8Y|?}#}ysk6YqTYeDKGVPB<?EU}Yz03DJl5%1U?GKQH`zw7Q^Wl8uzL->247;}JBzl>%B1_<alU5=--jF##!6aCtgNhDak-
PCG;6i0j-p;}jH;o?U9lEQAqLl17zO48DhGYag)AgL~K(wenC1dM01{j3&oG$v5GGr&u8DIx{lWxb$I4xKI+#9**eN0eb;ncZe))y$y@?!l2t{s<e3(B>YPZ
`uA}V{+6<mQzp^)`$nJ7z9J4s8fX)QQ#@eEl*_UB%?ZNo0**S-8hGD7dmw+&5l6k7&r5J7V?RHSOOT@y6juof&h@6@?N4pHnmBhfrF}Dc^EDAP0@6YuwmGnH
VyIggYs1m--YMn6y__S+=dZ{Lzyv98T)%D{F4ehd=wHf(yxO<3rU?=F@^YfELGA)Zbs6Q?pPOPSc&3U#Jr>2Z+_bTzve>ueV)*NO46Gba%~vl(U_)KhG?+IQ
+)dnOAI?Nzm96^vnO||pWO!!FsGEmw=ank^aVMlk{+xM5W{2X8&z096D53J`J)Xg1cKEj<cxA3C4(&hZ{|a>GK#y)BMUB`DkrD*wm0Pl~W@J&`Z&QHs8&oNt
0sbhyZFKvtaSWasGfj?5-bBR5m-PxFNqhRRCW~BiM`DwaC<R$4RLyoaa`*IAxkDU6`n?R#h0As9h6Ml1?+salap@mzXz?_x9IT1O@I}?)S#1{-q)TVb=NjX9
(dLL@o(n3s&!yKg#Uo!L1D^#Vu&jYQ6~TUZcdPb*WN#ckIz7=ibTbDT1oh_J4tY4$K(2poDGgVb@_7SvTtR4BWhp0~2<F76kq-L)&|jXqcULP6bpe#=ZoMJk
_22CXWA(;Jrg#(eTTeuUq{ms@j=`8^W+UTdAWrzro~xV)g^2%|=A)P0P`I@;ruNGle;#aj6df<br!ysWe+v>}NmyiH`*_dag;xjfQu(1edt=)<Rts!NtZsk)
>cgISMtpdWmuA8pR&7keu^iL7_Sz@|*$U=c0n2`{HFLlD;#&Zo+K=uKj`~7DE7{=ub8pa#bCRd)&tl5?RbGLJDNH-Pvk#L7;e~i^vis>^_?#{q`<ow$q3BDw
1zpadx2tyRZ&wGI(|hLJ+!%1O<xl1|#bZO*_okOu0anz%%gNj}LFK;2psm>m5Zo?jb8vLRMftHpTeBcsXW%exGYLU_<l`Qh&_cYO_*KOJMH0V~qLM3{v+&A^
%ZEAA4hIJdEF^B2Lb_&Su_`zc9z(6c$zIVY`JJ59S{DWqeac+d3^Pzg<&^{o2E#F&KRjQx7z@MwXU5AD;nme7e>N@xZ@SOZ7hY(`zXNNYPO5J~o?|+e!IllK
!wfm@<3%vaTjUxmibuA;PH&%<6Q1}i`5vaX0X*A-p6}*Bp&%j4A}t#FT9r168jj#1dz(GDV+PsNkBap*ORy<h)O|bf0u*;A`wK#nu`LnoAP{GYajvsBCzrCo
dva5WEu{laEe6eU4K--oT0!J}c@UDPI8h(+15<#s`TZ^jWWTPN`#a)`K6y_4X3Z3=l0Pq)+(|%-!}y~Fwn%7&Ug=-eYk^@!!duGeS~N+F^107mL7}ce*gXS(
9NTGg={g&Pa}F+MBo?k%e=jO=ye<xcW4uX=ePJjjVqtmnx)S43^mNfrf^cY{cm8g{d%SXgNMu~_0eJkTgA-;Ls4M*;=l>iPuTP$fvd@Os_yLIwMH86stL!5A
ppWg~Z~vTU1HpZK2`ETImewLk%iBmS{c10(s4YMYvqiy&yD|93<KgKVlL)=<w3~~3K`>ggcO)4tz+8q~3$<_2e?I*ooxnIv9*b9hZmfS&Z-?NKaB?O3r?3%x
n7=rd4Mq*QeH1@jFyz^NdTTooBN83NTmeZ4UA@Z<N^cBnmi@i@#2*qj$A3nvr=l{}=RsRTK3wyuQvZqBLX#y{Y(T{k!`m7;ubN`8p?8Ea{MrqOCEfTbb@46~
4oB!7M;L~M1M@ViJ&_$TDB4h}3W`7Z^tQFh;D6D0eUvK!agR<N+2<ICZwX|IWE?TLxb=qma)CXRG!+*1NChVY|6W13A&APh?)Xb3K$uoB%%3S3J?}l6gWThx
N*GdI`zIEc^Zs&LfAR-q5%+h-$pZMA@9Y2d)&akpg%u4>c!0S3vb1sGB~%~X)mUb?!`BJZ@^{yqaE39H;MgBSM3a##RuwCPe5?D$SFac}bAMiJC$s~-Y~#qG
s4xsRPIHpp@qofA)hR9pXXxA^9FjQc2(uWD`S1MJxH?WE()=L`3nQ6?uDvGM&)T9=CzpikgZ4*kx9l)l6}jMS8h`_iz5hPi1!I*fe%hKb1ur~XD;vV}F;MeG
MLsGC>clUi8NY_$s|Wp|yE(QvOio+*LMjZxH9dX2p+d-AD_g2~=Y#_XiWMkB!|)3a*=Q(u(NtLaGMvvA#m7b}TIuxhhwojU(x1IPs4Oc>8)SrrbmAuNKhAio
MRj4|njuIywy*vna)I_?N)92CbFh3me^B^lFjRf|y5mkng1#?8gPF?#?-&EPkDiJH<w8+*RG}wK&)fT})hOf9i%a77U$~)(HLc%YAPoO9&Yu(5%RR69rMP#W
L-3}A+qt9E3*@sFGv(VcI5O(gDB<Ua>oR%163;E*V|mxN{Fx7QUhdQEVX}ZizZV|GyCSwAa69C1FysbLyS&M91m`24lU}7^NLu^c#4zIu!H#LN{azX9YLu`3
PNocU4!Sbi97Bv%`sH#G`r^-o-{)QDL|9aAa<^Ehpe}j+9X(|M)DGT#9zyJcPG7n84=!eyeh^*XOca8a@^ga4{Ce0xaF5B=^B`pJ4F0|y0y}~)8ebhQz)9oe
{(le7<9CvwnJz~V?n(X?W|1=neZ~5(-G8AFExhNvSri5BkSyW4qxzVb@h&;smjb!obF<pp=fOKTy<cTV9?z^tcJ&+rG4(5!yg@n@3<_MCA>KMjQ>&Kdj`anb
2BER?dKfO~3KXVa_JglAce0_B6Mog}C=1a$p?Q&{wPP*<WKuTo10JZs_Lyaw8nqWPx6Vn8sr#bg#SQbN`TNKSr_(FWOaSXmvx7<Tc35uMDyiS=$GYr^F`GHg
AhwPoY{=9A6YZTle9M85|MTO}_^1P<8O)2Ek$@uy)h9jKRI%GY_(QrX3FHEVN0jz_sXF#-yyi^})|A#ew+h{G+?BCRFU=0}t<PkbeT5ME=D?-KQfHXv4p5}(
X24;Zvntfv6!e$7iLWb0A>{FVcOu(!P<+}xI6>`(Ev2M3@v)q}p4ybz(y)l9kA0*<N<6SQgJgzBvA8t3@cY@3U^Fi~lu|hd;q7Gs8zz>$9>B71%$jHruNsbS
lPnn{Ix@$~Z@3=K#T#96@`gBgWyklES1?%iPoAndUV<Gaij0f<(?P%7UCK=ng9{U!X#&yK;E`U<o2*YpVs7M9eL`m>n||GL)%1j9OMJ8<o$Ow(<(8a0ZwdXI
LzfJ;y-+{5V<%yl3$eT}C+|0&M<cc71<9Qv2$4($cbsy7ZrrtehUbnTND0Yje;<mx=NE>aHC{m2<JZv%?eAgnuCFBIsx|JhaSVpn`NEKxJh(g359Dc+7CYnp
xP9X6F6$J_fB7B%nW}^IdG=m{(yws+%!N?-#+kkYj_4m|XEhdZM;TRXAz`8|?$mte_sjExkJpn}kx5^aN~RD#&US*^Pr<jx{w09@NOjLxi5(=4l{cnyRblQs
&qw|#V-PMjzi@5}ghs4RJdH&$-0Z*KGtf-KcwM86p>RCZQ%yP)X0tKYQ#f2MABwoHiUvC#f6%<<-+pq(6Cdto7auxmf;ZoihdQ1*<6lig_~|WU_?_p8E$-4p
o5`nuWOi?4$((6wyq*C1(q6%)4;JW|`F_?*N)$!vf~xQKc-ZLg>2rh~T<{rrv~4R82v%=89#z5uJS7V9)4de~GJ))h*I~7=qG8DFI%|ubw$R9$U^iGZPCm%k
yB{$Ex^U*;Vm!Ov?vhY(3*H0|qxZj$M=_yl%W|A8=rZyjKkf8Df2NmQF|!%?I}4|Grj+rMWwWQ%M-PT`+bS1`FW}5(bmXMJ4Vp?SA2-B0U|XE=JwtdRv^0zr
-q2NGo8>*j7>zfi&3N=kSVK@>;`+P$l|2l(`H40R&mzHz<V9ltCA1%AU*&qH2ldKjyZy=P5QhrCs*D244iGXcm09oQ;bA#C*9@o%jeMcHVvk8B`_BBEDtPXd
Z>Ym;10B7UnI94k*xym@&2FOw;gJ5!TldURJP;Jy(qfPE4?nz_9}5RL>x+ytB3>AA?GbC1@rG3Mv3@7zeB9O^&tLr1h#zuBTs3<3===28`$)ATn8ZK6{`=Jl
cUr&LB+-Q6$I>Pb>rw+!bLt29-h_eDoz&PPH4wIhhQ7i#S)j{JyTfVWfKT7+sB6FK<Li%max4?kIF@eE_>-Ci!z<ajLOQ{St!0a)%aq5owW?s&hC9TwFDR*7
slqkepnm$6Eha0;QucHD<4yXiQYUXdI>Lr;xr^(7eENut)Hgjm5Pfn^<en1FPY%s9?DeT$gX?D8Bqe+MqiGxo*2mHWQKAQV0sfGQ^k|#pK(r@oa)vbv<JW(A
j7)0aVdmUdpL`!s@=%G9?6U*k)~)T$y*VmHuQqVS+8YHyBl*AMm2mNQU_K%HRcKd^G7=LNqQfofuJ6DV<QmDnJUkqRz^qW`-o5jzA{5n_-Sg3z^wq_vYaZx%
)9(<Idj%i7p9Om{<|FE0@u!Xu`~Q5(CS0>{iKGx3lEF>U%;}J!E~jL+vIFNt$0#|m2TZK>zbjqv$LZ5il$sNM7#*;UY2wSnPp117pP1?qAUZlC`K}bE>*ir^
K6oP4HvDMZgE+7h@Ufaa(uLU5ubKUSQ}A7pY{!Bo0y9q@J2sHof%Z)ON>ra6<`Ry$J)~*E4SBwZBWWdY{l*~_A#8{?^-~9<8oUu*NFS19Yl&+onNMDsa|DN-
TLih1B;IU9e)*B^h56LWr?z8tQ9By>=16c2zL$BGhI0Dg-gEY%fe*T%4fq*wa>@ioZ8v!SZ95}qeaCmmTLCu%-y9TWv%=|itNjU|^k5Upt>Rnghmq7#o6iaw
*q`t8+3>vr<o$DA52#wf|Idt>g_A0R&HaX7=kE2^W%GWP6)P}|zq(a_!38oG3Zr(vm_nvp#{XlNI?NA7+sxYWV6rIwMe@1<uGY<Ma9HVKjV^lRq=+L(Gk8YU
&-tO<eA$&qLJila7k|*0$l-|g$#;oMCa7Dybd%$Y38+55%=p*uzQ@x)>bd<QVcXs`x*cl=+kI|3XM)XOINhZ?n&JrR)k}ube){O03;m}V!h<SunJ7{xM`*`V
FX^4Lfwi;pk7F5h=qV#(A7E02mb&@<0%rwGs&8Mrz8QmP>zn463j7iBpjkuiXe3ClYAnri*x?VupT>@eEJOt=OX!#cz^dA&dHGf{WR6cM$z1lp%I${NY4zIR
?mY2)!PE%B8`S;7$$r?Nq`f7ZrjBn%p6;yK>jK;_*l*kVBeC!?>$%hBkX53Xa(ZVEb2iyaB?sLQntGR-?70z4GS8e@yI~JOo>@{yeFqro80yXZ3`Jr+t8lJC
G;S?-`==UvBkqYZix7b!7=ljpO)jb8@}}(ZZy)VYdZ2vYCZ#WE?+Gsn9x}(LoD*yuwgw<eiyL1lwnxU_<+~GeO4z+d-f{er10;tim`_kdf=-o%^`fc+1Z%^o
CK^l;;-hze>XR=Xm@--Iqcp-?!2WWAs{wGeICF4WFa%Ub6h(CMoiT7_^6v7>z5Y|`qs^QyjZ|_b@~(O-7|s%*>x%=HBlIIS2d$CRUMSk8=njhagr;RzWg%8R
apaGsE!I7MZGRD#2lFL~DDP`$5IuQHUufP5h1QkDF<hP~J@MfZ+msu|cTZ<%&+DP+D0k`oa0^(G3I8kBEyPmA5%27`+TbNj{vaz~fY#LC%l^y(n4?)ay|ZGB
&Ux!1`7LvhQ60Z7@%Ah#9gV8q%m{!@kK;O{ya|L#d4|8QBtovHipVL-7G-r}^x|RWP&WSjjH^lqDL08_mq%1VwRqcZ{_{DgX9j5*RT+RKI{t^vFH>Aslx0-s
@IzBZb#-Hj0-hyo#~$FW!CfBlzL7Z-kk%JAyJ*HDnBk4g;hU!r-F1WejrBR$UaLJamtl*)*@H~mrDvh0S*NWg>yL`LO`E7HIi$R7)q6;#iNjNxSve~i*wT45
NW7|tMe6%;8IQWbaCTAab&4@OW{rxNr#$ey#%fCHo(UZ8oO1cVt%>VGQ8{DvRXEgqNcy3nKIUIcmC>1+V(#Hp|7+?ruzp;1$40~+Y{#{{mZA*sP3xY>daxaw
e)d$GG}xhGg4?IhL=7BMrK1a^N!agxONE5e7LjiLzs}T&f%M~1w;eS<5aObT(rY7dGuF=EA#g{{%EOaFBW^en!{fMFYl;sWRl>a41}Iv8X<~BR8+Yio8x+?i
p+*+(bic!KPe;Lz1i9m3EFmY&XsCuxL2eoL{Vph33shLRABec0yF$IAHW*#}e)6e>9v=C<Fw$i?^k05w*>Bvi4tIxUla=0Iad~Jx+is}hR>G+Z3n7NaVdy@*
s_P_Xjs1VD_I<gfjn5y>dwUyM!-I?^dit#n%5P}Oe-7}+M(ztIt38hV^o~kVRmlK)K8h0V1;#j@l;QGr!4b&@ll5fp4IvfXzv~>W32x8)QniU%NHD*9r{?Sl
l|Gl}bHBV0?$vVq!Bz?;l>=T&_B(^j_SNcSlO)K6E6A#6@=+gYD5_H*k8e)J$rJD0arSu~rFgvts*5U!%)<TQ(%{gl+!u$;din+hZZ!x^9%T?*5rwr=H(lK+
4!mAdV3rG4hNh+2(b^e*3^at7tK2uk^THctxlUrpm9`XoIq!-CY=ksTmL}kTnU)f>t`1W;%^B@|Pd1(LsQv*_#PPn^`FhwE>9VblNILAHy)H(uds`OlyYe21
oNDlDlW0pjrH*(n3(FT`g1Ard^-xxwD5&4A?y7u~M})_S_4i?Fh*~auv<h{_<9t!~Tj^pLzkA7LBi;i)_cOopH_`(!?ejXRB|TgWW)+vW7r^F`;|C37gRqa=
I?C&qJ)Y($ti*6+;zN~c(zz%l9GUP?>{{@|vS@m`g27&2XP-$9X4c1pZ`7&5_D0ZWb?~_HTnEB^6kAMteQzl>`Sk9u322*Ie{>=NOutAF|A~;oKZ4+QQ+Hj^
Mts(6oL&(Qp*}`eIm{6L^5(iZqd00@2p9Cn1VQ}b<?^?q{t$h9X(RZUJj4&YHm#^LN6(pz<>Dw_Bz9$Sh|{WqFm9QWlGPZ_*;^;3yLC`E==S3jo8;bHuL`OA
sfhRULKv|VgSt^`L)$ks45eHfCHkU>2q(@n?ukZdvFoh(pkf5!8y8=S`Ksb=>16q#abfHj5fvXlE(cBh6K1S$v|#Q|dqF1L5UH#QZk8#M`0S*$ujZy2EK*%<
eWICgbBB%D-pUYHIi(-8QpiDg?xvpMbv8JiC-`>mm<g=%j?+cIFoH^AK`!gI6)KJSn)df`L3U{Wqr4Zo$jwmeHq6q%#B+=A&jq4*QyRdxA*q3Vfx6qKwEEz=
*JK^bY7Upx_iZV%N>HB+71=+ng83b<u7a2tIF=+a#vIc^GqE2@j%*w{a!U@mpLB)wf!inB#^fNYl~_yWsRdSL(xK`B{r`MA=;yLyD69t3IJe<dWi>oaW#Odz
r;0owl#=vFqBXgd@6?`7cn9^4KPA$^*=w`k-<T^SoTE=8d2fzwy)7hVaZtjBW%lv2@6SWGB4se-l02fNqds&`^CMOJ2>G=`YLHF}{e8eT0l(&G3{0#P_qcSH
NzzjojW0^Ohp*bAMZ(^md#``12rZ~88=u3I!L%c_3;~E**RczbFoor1t%ot0S}5?*d3T0b1B=mrw9E|y&|3F>nf_)uvYQU%(r5rr&0DQLG@8NWHfP%LuLckk
ugD`4w?<v|MA&Ay5&XU$xpkh@7hd6RIZE30IDOIN$!pICIIF#p5s(syH%6_$gs+yM)?85RzGDI+F7$o*XKRg~;!*34bw?bHOtcp(F~wv;VBBVH7}CxiyTdPP
hlCD`eUW?fq^2$VOwg?cm@ZX4q)|7<PE>%UuWu2&YhMKX^+|(vVV#!1H77i%(kgsB8-#S5_c2?YP7rk~IFzoF1_9gL1Ao4#06Mhs^Xz`0d*}V65<$4S8kE$j
>ILB?(m)l?Y)HQ=x%`o1@Anfqqkg<e1@}*-2r++S1ij#XXSf;#mDfK;7%zArP4w~3#a9tH<(MF_dpiNE_2Jy!M2XnHeZ&t9BB(kj=_g8Kh|$>+ZNsxha5pY3
xk0LfDGTLHo-Ju~CK&fp&UxX6;*%3uewt7)xTRW7S%~fXioR#^6LEASdOg;)2&2|7IHPm|AVKMrX#DdE?6#a9@>c~ypv5s!q5T2|$pYI(lrz9!Dna_Tu?|sM
C4X+8&%n2t&HA1IFX+lEyh<q2!%E_5$>=Y8bJONhE{R|e{Mi2phNoY|-*=veT4XJ-QFM5nC_ENxY9s!?*n-hy@|Nm(r3Q{h2lBZN7(<AL>$1Ck8m2E^W^MJh
#Z$6o_PVbkpnkr$pLi{lt`#N8DJASZ-`ICiB1(Ha8a(|%QWVAo&6*xGKB#DMuN-JE!_=0_VZCZ&@CvQ09e-j0`f7!g8#aFUP@1UL?5vN3v)XxGlO~vPak4(I
;0-yRyl)+xKvP`)ckP27(0x#)lbxaquMoBKW+(j->~QTOF>O4YZXftot|0@$x~@ac1aA0;wFytHDr|nCFR`0R#e2EAUkNMvxKHx^ZTEc>yq_S-{+^}@;^#HP
yQCf<|0cdgSY(96x#+_y%bM_A6MQ|gr<;Z$)WjMXVZ6Oq{t=rIPCCX_e&KY*gnpOm@?#aqKO3qMmpOr2OZBbK^fV|6ut~d00PG{VcFW4a5pJIbCi%q-Ve5N~
=z{NMH1{fnWp<WeIN*$1%WpFbFUc$aq{#r+n7&-)hB9t4PZ<b*kj0-gP6nHLg*|?y?jJK%MQ)UY#gQ&U1oqZc=2BZALpT1|uf6%{bArU-p9~WmZ`^40x7NgW
U!ha)?c>3rIUB_2b`l(xlXqO!J#c{b_GhuDWZ-6arv6BlANH@8nz!F7!jCoOT{zJR7(`?a33!U&&x)SLMy(;TXp6FD#;q`<+rgI+B8E53s%zbymQa1Wzu?o=
v$!KGFJa;vfgMhN|6A4$@T?l?Q>FYZH#oS%Tw$1u!)*f3{5WhuJ<%T2{7wohw-~kbZiV1i`JvWNB|<n?;atBSYlLshM}HP(xq^=DiF(W64AQROUk>t#18c7C
v)@vNSUR7!Hps4rLBslIymySy|8aN!t7JJ0RXsSNZ=sK%!xg2MVnz12Lo!b7mK!{Ywhv#*K97_@7N=e<6~t`@GA;24;mlIL@eMCqoY8+a@z7rYij-;BPo32U
%cDwPp^NhHog1vZAGY{keqUACSRiHy0@L^8r-PlcxZe6n@rQ~ca%sE#<xi`^=x>Ib?D0L$O=xUg_cKMqf~{kYa6Ss=57Pf|ibvb$p2p+*efQ4Q;rop-Npvm`
aaB43C)RS?>8@x&TXf2EcRCfewe=4zSkA+#P$S5%Q2;aPjz8KRjIm@WI?%Q3kL}mVDx`fXD7y2K<KuE1y6V=Oeq7p{6E+%DOSUYKN?u|Rp!bF_UGdf1n~Ctq
9xVQZ^N2rv_RGk)5PZXUP0S1U@yv>X_)wo4th{Wu+1GioBXEUqT|EX_G8t~0*9Z}BFRa<##tut~BccT+et4#n{x$F6S?tPv=RX!Njxo|3{HB(&FlG6W;Be3d
2`Z<~_uS)0v8^i6EpZ``zjFyaSjvk2+R8Zb6hpk$9Mnq>(?+K9cx(K=^PrfD6^{!rgND%;=6+UTSl?q_a9L$TL{L`kqk95qdf`udF`OAi`&71k4pJgD@z7?(
MNNq5U9neM;fAinlF8Abz4Pv+4-KOi2kYhAuc=ps;C!2M|Bre)^q-w8nANp~>qOgH3X3Ypevs!@{W^!}W{ypMKN0A+-D8h5(T8i(=;JXBL1=Cl8qvOyhaM->
ZOw;DIRCc%0b!{qoT>+3B^R7U9UGSw^Mn9?gwZ<j#4uy>QmogXb}l@e)oJ!Vsfnaht+#A$bK^HzwKqAZ*j}&PsCKH502xVKgbc3)jQ&iOPc0aL@nenn-C+hi
<}NH`IHZ8+q|TTRvg~Miz}67Ud=_u77fQ?c3S;bLTonhMDA-BXsz|MQAR(g?K5ZlehU>(~UhT4?x-f3&mzWR^JgyXH?d3)+pZ{aca%R*wozZECS3?K)qw1^C
LP)wU>&V4J2fba2r+r3jpbWPNxJe_9N2He(at0MpWqoIAJ&6vfLl0c8Y>6Pwm*S3r7Zqq<A8~7)^}swoqejoGa}YE1Z|Yd3fRswe`}Y@=pdw@8&>^dZPYH^b
*>rXBI!!9#ZUGf!k9DpME{MU9H#_;yGd&!1`WWgLeHyplHfYRkvq4Ln&WP=xFk*f6`gnfP;qol|vHiVjIKAI7{7(QM{30`NO}gn|?3q5*u!$tZQ#z)GckJ=8
i#RYcTm;0~Cw(3oaKf>XpEI~s0q$-E?c<}I|M|3zOF}!aTmt&Hw1Vcoi9nvzML(fl2)tS)mtR<&L6)QS_9uO5khl^jruZ{Mj4O?*l9mro<jM)+dkr!4fY-^I
S`7m1H>`#rh^p3oon()t(4Of}`sl(51dbnbE&U;an%DU)-CWiv3rq|C?!pCfp^{_Qq*xGHx@nTupbVx@mpQ^%IpI;esvub}fa@h|uXXhNk^RER=g=!n-1;{A
c%48UQZ%9Frx-+FIAU|<M7k&3eqOm6Jz$TtW)pr15=|5}(4HMpazfX!3B5Kd0aWSFjJ{7d#)G_H`SBbs;52PwWYZOc#bpKe7Xl6-Vf;F&{ArIzj%ijh83mvr
ouE_qwKGV%GM`9ITZ7#C+3bhY{&=w>yYE_l5O}69i?8XY;ID@7KfcH^xD!i6ss-C3nQ`Xs%>_TaJ3V-b>9I3x$6m)%hAF^2CsQtt(H(`_{r4|l$iP8k@j8o0
J6LvpCKSkxMb><yp18OnTIT#DFaPjHO4YF^M)YDR5b$R5KCqWd3dVs(=D|??s9Vfun}^ci1K-ZD1!3Kqu#V=5E*>9y@84MC22qo#FkVM>Fiq~B8<bPP55>=g
u8*y7YWL3x5?(9F%Mo*aU{?df5O-h|t1kLn4_^Gi&5Pd6<dDw~1YsO%*O{|<0TGAVSm~A>(9CH-B_nQvcb)}5o1a(V(#j~Ix}OPT>KjLhZ>B-7pXBbVCJ*fU
{^3*&aTt1zonnr;ZjZD4iRv!-UWh7h(lScd!SLvUbsoPma?6Fkk6d*G-Q3sZU+3~L`19b9H<2IG{LT1t%<Un4x{2|;g&~9`^CDJLz3}i`Rh~Xc3fc*1%ztl&
z>`pp_m6ZA3@zBaoT=S0bWGsPwfXaCNJ(PP)-Z*Bp28hnc4K(nrzO7YlZVaS8x`&kfdh6v)8eZ^Sa>*k&4Sz*KCH<UgGGMuz8R6kWFi5Xvm>ppt!7{fax~bQ
2!`w1Kc-gmwqV$29wAAP4d&CvvD8oX@W4lT+2=_Hu4{ceU{MeUM$($P&MSt<J!KtKl&*>0W}{DSdmQlkd%YfSq7zbn8QecPZ~;{Vu0FavT4;-kePuXpk6wSV
o7!z>K`8$5uTiohO5!dV(mGk8zT>;cMu8frzqoMmUQs}rqzftO8)amKxJK&kacH~8llo>4b7VA32uazSLFIYyGmGomSd%IANKA1;D_`E{i&rJFeAkz7Ka(Wx
c0Z`rh&M%`pe1RO69+2uXCjxr5#VIN_x69)yK<^qR+CDVEGSj$rjap-LEFpbyPrM?<Es2FheLrMY)lx08y;)IYtZL;wTUrU=>^Z;nJ|XftjYzwJ2Hrw;_|os
YXZem#f0Ivtb4kK`WAx#4)plwOkC4K(Z4$c8tjJP`upJa?o1&Be{wetpF0W%v(^4ocX7BK|KJwm=!VP6KiSmN2*JSC{A<Y12;Y?ttx{_5$kD`;dafi>!~Q9S
AXgs?44*Hgxc#8SaEeg&0|7zEC&h99VNk&s*X<U@1s71WSINZRqrmfgkB*Cf72w%xW6cxJxyMQ8l{+o>^#7rQa^Fv7q$;L7p)z<U=kmDYK)}5K5D`8la<f#0
D{)7h#36QE^801Qo@0-}-?zps=4o;Cji_)Pr!t7_Vg*;;>!IP%!Ivb3!myP-NBa9^HBLEG{!@!qK_JnTLsjru&|K8Dud5V-f~Pvoj+GG((q$`GZHS>%`<nB`
QhEfWt|~uN-qSl#6R)1AF$$k66feF}!n1BdvV)<g5qtV|@`>M$IDWuzzqvm(oWD?{wg#x6xHhw*Kv^6oFxlo??Dk)N@4LVGjKM(&LX%H5Y55g!m^Fu*`lKGh
^7k{1^1Fe?dHat8xjSCnpVm}k)5edbZx_zQ?C~t|(`VI1P8gfzu`{%{M2JD?KveBtx$AaP_Z513FjzLPaiv}c<UU0nx$ah2;tsj@&@T_wJ+1XK1T|obKh$;X
Q4G3zpJggWCLx_V-2Him1!M+IoC9h+F_)imtaV)<8PoUXPPOYpBylYFg}phtw(U#=GPOZdq^aK<EsewK(nW9B9r1w1MROy(3>|FTht9t`k7+L{20bNtT+<9<
%D+_%LGQr6&!4CvPa+!J_tqCArZzX%ib6oBp*iQ1=@0Hr8_Sw!vEbw5%8Ojo2Q%I48OpMFJa5WA?z+zpl+UQ-X1i~rNx^-CR{ts1E8WATzoz5JMYgmu6({^H
;p%_sW{j(5IfoT_&+na&_sC0mX^6`aC1hzDB1f<@NK7CW`MzUB*Qbq96eM9(5^9M;zZ}L2$wW{tH*@<<cc8fS<rlt-))*6hS<<W*iS3EE-xQvhfPd+fXT}o&
kW$DpZIWC>m3y*wO|c2~NfZ436efp(773R2bsLx{{nR-oW(@Yx6V0g$*Kua+-+;J;JEUgX7kKtKE2?;FXL-W_1U=UnZ`U{@yM&{5D0DBUtExWV_^gIf{`{ER
G<HzsuHC)27z9~G%?A=5iRkMeEAyr^M(FIv8!0N4xbXLDc`x5H{5de$K0oRSKK@_FJQVHl@+z0i-NOdB<?MF-qFW$j6(|l&D`(-i%AE?g1%Es?P`LPK)*0=-
Hp4CI^dP-3eZQr}7<N8BmE8??;PbVUqu#Q|Lb3?QO>z$KaQ29P>UKr@-z#^1E6BolE!?v)I{-n>r*<L|#Bd@rN0>I)47-AD%!L2@zV)?`k0q}!Vb#gMQ6)zn
wX;NFpX!Z4vDVfx{YMnTCJA@SF6SegsQBW#Ob%kjl-h&4mEfc*t#D>H6=j2w)MO;(7?S3ZAUm0a3tv($6_BXn&p~(nu8&@r_PFoLU1kksu1o{E#d7qmRi&tm
hQRcE)PTvF8rCG}DrgICLZhDNi|Ctt?3-HCoCr-oPeOj%QcgG;?#+cKz72&(h{C(>%Zf0IBgjv7_Q37=#$Qh34iMrnkO-Dc0_8+gak`S>e?F~tPW(non+W6j
(SMPgNuXJ9`cYJ2g=Ej36YinbI9ox}5Fg_S-mw;CoC|_<Pwm9!S`I9qeLGWpG7V`GOj&v+nb@`OygF^U*ZZ!jc@<~|qrP%`c`HB<w!4o?v|?j%lt?|%fY%Ry
`_E{w?CIHR)ui?4>>e+QQ12tFD8)xY>Hv4KX#ACnKdYptgnKbpZK%D&abuA8i+}NXjQM2sI*=Le%^L@09|<0OJ5A(p+s+i^Bk9AnvHJMgtp9PZ2!S~GS-iE8
HtbJH_vOY*qV|r_Xv4ZWKHqRsdo5y$QzuzW{fDF=5yVQv-f<o}tz?C@r*vR&pm18i<s3c{NPFpf$Y7z>=V?Wk47&Ox?y>z)!q<TsyGB}ld?+s`RA<pcILUtB
^v8xM;m)!u7`H^i5$5nBCoR0lIKC>_A&GxEcTQ<LGUMrviGA@uW!z_OXj!?U3IjQfvF~4WaY=6Ggwuo{YV+Swxn+vr-_bmaHx3er%kqDJph5`|(u(S)1%@b$
B0Vsxri-%a$D_l>hDa4|CngwHMzh}e<)^yxV7op``E$|+M?Nx;i>(3^uMc!D1hRqUw#+TdA9hHlGOh|#QU?8HgW{!n0$eu?J#qMl805bY&QU}HE9uJL4_yic
CCB&a<Kj9fDL*kzaXT4P15||M1CHPukiRG_<_emfqwhcL#=%steD3gU7;64$_s>$;;#rGxwaFDpOf41_uahcZ>y8SW)qp1$7`Hw)?(zE^#f;Jhza0A3RYc3n
^zn|NH-h4%3QoSEaP?Nx14H?)g~)w#*icSfV3IL_S^n<JkVh(TtCX#=mNUScV^>ao3G_y}aWdf}3KNJcmbPEAa)b6i7f)WTz216v%^-Nc0vLU-Tt0r#5c=^p
TSP?mFjR0(Z&%fZl=rFoTh983`t$PYMz1nX4XktrS%@K)K69Da(hhg5{eFBoS%5>soaE8Ywm5R-*?wBlJ%4vUm^f-@fV-vMCzFn<Kt4a1c=ehF;$z;aEYll<
oZ@6{7qc<M3C#kCqI58GS;(B>Gbc``I>{0~bwUsQ8|fx?UHA}wxnlQF0~y-0v*}7^ICfpVCZz8a&Z)WD6Fe}(yC5;%1wkJ4pDX;L<U)=~+uvQ@JjUqwtezO9
?Sfd7DA|B~8GL{7vZF~z4kvrmxLY-~ASug8pIMXwQxR1~TW=j~4F>b3hWSHq@reIGSP)F?qzG*fYG6iW#5X;~6awb9Qy&KeFwbVoKy+LP$4V-n+awy|TjJ7$
Lzyn#g>SfcZrg*NAxEEY&=5LyDNQ$ytHS2m`so`*;>f1B*Bzk)9B=HG*q5k@uNP0sN<Ec@)LM}7a}O(wHL(6Rrr`qZ<r$9I&pGflZt0!4X@bu`FBy*PI)fs-
jfs1l5+ue+u01yLa6Wi!Rk%|TD^%Ld+I7M(KGM5IRTY4abdn!jGve^{`)TZZS`7Q1v^_6M^M^(;^FggoRxoojq+*R91#QX!66FY0G!8`G`}AT@hwYy-_dPPh
L6ZE%LQ4Y#zp)*5)C~q}*S&k9o9ajiO4O9L(nrN<!WX70hoMOzqhh0}h8ocVJt`6f@Ud=}lsjmHV^}zxk6Z`06@s`DZtn5*>O1QXx`j9^?fo)-*#MNZwp8zl
84#B%LibHt9!e+2PEGcUqL3~4p6Z7HZ1dF`z2vim6T@ang0nI{eeH`L;MK&*=LG&Vul&KWFgdro*GJ!+*m<INPzT<#B}EdCig7wg>i*~*1N5A9eV?UZ26m=5
sxwp`P+A}nVWTvH0sXri&8r5`3VG7H_;o|>|ML4{*M+Y?S>4d`wlGCKLmJJG*L#>;#6jP5gr{B10s|jku`I1S!R%`5C~2HBUO&0&D#ERe=Yu-)?0fz@G^a-y
&71_wI|hTvH5N!cRCup$pEr)R5VkH2#z4~Grk~YHI2gMkc&JQ6ARKV4Pj)&3h4j2E+C;v{p3005K@fE7Z@3Y?3PP4GV{dSwJ=Q&P#Ijz*BJZ$jQ_oEoeDRID
zfUR@hj-F<vLXV}#^lB!eJK^*)}FF5)WP`pnf-7lQ5+tMY<fOt3xJ{D;#*xQ7mOG8H+t>)(REA7_qx(K%#B>n4tg8|yD_TdoaIowe?Yp~EENK-D|~B@Y@G1w
jMIys2NG~@uA2M9q95vrdHj=y0@0_(#8m#$1nqqy#Vw=f5Vyak|Ktx>kbZRj^~Nj)4w<rN*q>ZLLXmH#$qOscT>Eucw)Z?{hAz#l`)9!@awu<nBNS_q0&FsZ
A$Zz+bDX146+)Lx>BRbcachm|H<N`q=K8AUuUrWOr8>j9Y>zi|Z)qkt)cPRO_25tCMp^uNfB%*8jbLc8RTY02_6LQ0Q2LN~9Ga#IN+mxU!hdr9M5XXvzhrIe
v^o(23YRvyecg)~^Pi$Cxe<+DX8#`LkR@U3=v&GXx&YjamiRSh>V^Jhb2<Xv3|wZqlA)j#3pX>b`kT2KSl~an@g~*|rYc$$hra~w<!IqCR>1)5OX_ZZ7@-8i
`0i`8W=ZH+BOR(-35Ln2+KKsJkr?jMp?ol7iF3hMG;XuRBlbjS$r-me@OL-1z2{2<#Zth{EkR?{{>^*yJvk9xJA`qWUAa)V4w|zF$%X8bjJ|6Xy|Bq(=@qJQ
!4Ugt$)a;t5W`*e)+gKvT>Bd?DN(&dtAyjJ-T>ddyq+o#Lo9@uevJzXMS@<v@vK%?JSOhRCiW{t!JkjIL6R^CDoFvGMfVF(=E;(}pFaxkm(P|Ju{$G$+?a8H
a}YEh{q-2IiNdj3(XQ6Zf#7mkrFwqN2s$YhZOc#m(XVS0G`!a%FPOSndNl;&j0Mkowem#FxZ38oz4iceyUD|JCS5rCT6~&}3xekOs3Kij2F%mHwNlu|qKw_z
;@?yvg2PT)5n5-#*<tDRRkl#5Td<LJ%R3-(_c!Jr`+?xJ`zM>d9sQqASI?<!M5yfPxm|N9N;@4oI=|BI^>jf!Z9ZC#I20_Q=Hkx=jp3keKC{si3BB~QH}XE4
A*kx#viFt?I7AN5N|OhJMnUxTqm_JwDCVvWAOO-KE`CHuO|dw(lr3rE26`hqaqUx?u=Hn=tRu+7)Z-b~*e|wliFiPfb~FfQ*j?wl``z$a$p7fXyb<osR-Z31
O+-YG$PW*W6lB$HA1_<=L;X!<Ek1W=2o8N1;VyMY9`B9PQx1A~!WSb^tYU<oQ;Or&zl}knb5}1eqY&KJ`j!G(+>t&xwo4cs0?F0LzMIp5m`-f4f6yBMd-<~P
@N04SQ!3_6#Nh|o)zi+&tMP~mvm#vMb3w{WM(szcepn0(v<>J}#^3W74m6+F$Eq~bW6oqRbh213#q3z%?pO$auU{>w2U5~+CAxrU_k_h$6%TCKIw{zB6oYjB
$*IK)PS8GlW#FGo0>mskV)<A6K`ZRXrd8k$dD)DW-%Q@vxi<6Q_Ptnm_mXazEXIP@HsW`QP$~{RKJX%SX)k~8Ppc}4#bePt*HHDbFT!tGI9Sx!A}LP2O-V8s
PjC6%q;nR*b+TvTo=J9~c&#!wd_NW}13m%*y23cYTK9wFgd@H!96`n%ZCsCk$o<7E4|(;AM+el6VMCvB|9Q0wHdUXlClDE8K5^l2@vIq)o_2QG$tr_MFq?f)
GY}Q*A7rj7C4%Y3et(UdPPoGP`KQdI1RQ3OWNS}!!2DG09LsfeRMSbF8@TKRm0S6b4CLZr%J}LYTbwVJzXaTWCFlrx?tOt$@BDDjkK)lel^o<Tth&vrdca;l
&;B-{F?7#JM0)1<A@}81`d3uu*zI2W;Xmz#u8C3780}1uC%4^}W;DXAhopoXnKyI@KD>WlV}pr0DnWt9Sd?fsn0QGUVENCnW%2?8>^Kix*~xR;;}#Qw-FSaw
_%a*~Vo?CE+y1c^9pUJ5Hmc>h8w<NmamJ@c_7HUY?M^qGj4OPHtBL~b@!&;^Pt&F?s6q(83}5!aF|OfyC&x$>&-b*HbDQG!GOw}!B{!TP+F~#+v&1yh$vPsn
Q2gnq`j%8&f@Y~(=H?GgLBc#e%}gHzKAsS+7ByQqs=s<2l5T)-wx%FA@(_%whSB`3iGY5>^+!+ERFSgw?DzS-`_<MMe`TB#iz0Fxa_>lQWL>S={OIY8&@ktI
IW0@bO@%)08Fhs0w_dvVpg;(06CHdc6AwZTs?q+ZI#3gBG1Sqs0M&=_cvhtgkR14Vh~eH|Uj9@H_s~qlPf7hV>|{O&mp1sWrmqG{{@x!;`-0%H(n(Hs_<#TZ
67e#@Gyo#zKM$XJw8uGrhuGUDd_gFgb8)^d21hfm)TL+mg15}2vOUorA%mY&D(YRKad%|){7GR{e6ca0D0V=^XL6g7j0CV98@G7Wc@{$oKI6mzHHapXr9JfC
9$CS+o(n(o$9>%ua<gd{^tUXmHcsi`<s?s^>kCbUOnlZA{1Fb(Z*osQS@~mggFWE#NIZT!H}kA(ML@;$K&Gt9IS7$llopI~htYi#$MCQ~Xr>ifjqT&c1wJkl
=Y?48n2FCV5F|sUk>>sX5OG#vQT1;ZRs=;*36W4xKvD$>X<^ab-QC^Y-Q7wHqM#xrB~lh50tN^Qill`S7AW6*C-47s4rccJ=6Tk-?`!XC5*u9b^LoJXSsFs_
g#Vz}?`e$N={$qZebM;Maq8gnNIPUcaJc!FOB<9wT+?P_6d@-bR2|M@gW}JjshrBD2s_v593T@9r@%E<6<Qk*KiK2&@2?X+Yp7feYtRRS*IEQ4aTI8`%VmEt
`M{N{IHq4P2K-0<Zm7E1AUlV<?k%?oT4JPs4<=Y6a_Ws^Nrg8`ZE5UEy@QazN}=Jk!TVo+8=BD<o_-t-jf~vqX?I^?<dpSi;b<p}3fit%^qFAq^@a=6$IKy~
`E2{lIHA{zeIHd772=s*{i7_;IFzZ*{Tlxii*Bp($-l{lP<)v#(r@aHueK)E{iJs2Uo)bxI%NsNgtuB-Gd74|5xB?V;0cfRUD*|hG>9+F?&hoc;byr)vyhAf
o{c3-3#)76)yBGXT&NRXd0lB;$Rha9>ip{iVvf+6E#}lQu)*fKsgmn8A5^)BSf1N7K`j{rMbu$?yj>H$^WM@9W-bnWjIXsJct?Ysu~ZnP1AaOI1W)_p&-yPc
&<{oJuL^#Bw*z0CdG(7LSCCDF{xbS1i=6)Jd>KCB;5)baInpZ^N#|WRco}T4ubPDAa-0&%#KkGhiA5oOcrZgD)&ga|g1>uh1JQZK#%ptrKU_JEJn!AqL!HA{
$rN=3q~$znN!26trs-d{<|%s&456rA+Y%81`@0u<1o1+efn=jZ6_M-iFL>^#z>JOk+RZh4$UanC#DqQ^6x#GHVjbWsDBx36DGq-ve(J~TDj>`L=wM4~2B8uI
yPz-^Z1jvo|D%<J+sPBbMkGmic17Y~La7s!Zq&AQrP$y=oMHiWsWq1Scc=CTyP?H=_NHFHKjPfktKA4a(A^~P<<<{3Y#GtF+C8?$7fP2e%W#9J!;#aB)xOBf
$}<XH_rOrELGR@lbDZD&y+CWqkHhIzJq>5wKra2gXPYDpf7mpVIyFqNzNX{L^Fs}@4357Ov~9t}GNl0-U#wq!lQv*x2lokUkI~a&c%v5B-%vyFYLbg@h-0JB
y?AsdxW)~KD&DdB*vW&l=Z@wna$TgnBJ=E1wL`BV(X*zjdI*Zj`OQ>D@Y=i9FLHmFVB}=b(g}521W;(0S6H~=sN@NIkFySFAvY@O$hJeOm=>StkS-`%UOx?c
=8m6@reE{66(PdY(Jk3yh+ff!byfaATz;W#7j18iy(1sFi{F}|oA}VR^UDN5eAadCC*fR3Z5vO1an;4`4OQ=fvzn;x(l-|}^?-g=h4s}II~?rsbrU8NhjD10
{uYxVkY_e>uEPs7-yTvIFoxq-uVujCb5G>^zNLFwZiVk<VMAx~0--$jtFSoI2DW~s=^sn2Fmh13I=uRS&%q~uBae73VujroI<!+vo{+m(D|#);9%O@|*I#dh
Kww>*w)dkd45Fou*sM9=opD$GQ#KuVlHL2VIjaYm+RI<I+f5Km>M|No5rN2(yRNf>`mn5=QdQ7V#@S_Ica0Nj@QIS9;M(s66Ca<)t|_iCB{I1uzpa93j#ppe
rES2ZYa&QKuY&tASI-pi141dghXRlSYc|tdCO$`K2zy_ZZHd5<#eWyFLe1g!QY~sA!UY(VHQDh~L}0sof~}t(xIXq-#N}$E_!hrn`lldVp!;nVp=*nNY0d1P
=GKTGVzqzO;)J0kndu8wdI&D&VBt9Dh5oJvZJ*PGb89{O!RUf3*3Mp;6)3mEiOxG;6a_TV@8GyoEF=dumqT}hC`3U?FQvLv8;pI-zkXi5Cl7<aj%qPu?ii*L
`~Aw_9c<5P{*HI6;n|7IMTW0Fm~*<`DX?meuYD8MQO5|}S#<T`g`>KVj~pbq;ctb~mA5kde9oYF>c$e?Lg1KVpDuqS<UkL1t9u~9%hWq)i098-1ly_diDTuq
kaSY;@+LDv=J1KB&TUg1I#6w~dCw51V~rEXB`=_j`PjgpH<|=Kxn*_lvjt-EtVr{>g}}M5jpdAmGk&nM<#BAwV0eK|&)Lrzt)&tD`Nw6UCG|<Maz-8#l7i>P
fAe9vpmH>!T^FXK+1Xv6dC^Iv!oz3p2*u}y!aXw9XnvuW)J5QUQhj2cF*gfjR4tqma|;E7OzUIPPBHYfQ(RWeaDnN7m+-W?H^Q$Twg_&ug9+)2tOh<EFx4ww
7pC&Yzqt{%o*f(5Q1Km3WYWX?wS8q(hF0J-u#oQwvcu%HQ$iK&%J{{x@0x0#19l9<D>h?XV70|xQ%A@{7nu(t0a0o=N@=;!vFU^RjgJN@iU?d{{Jo^pofEaG
Y0A$Q0AY#PxxzVX5V^3Maod~Xq)PM&KlU?-BR|oTmg$CfRFYipmW{BNX*%W!q0hz0cSQw0+Tkyql~QD>ID(Qb{OTv=pg=O=QFcxf5@v1YL+sXgPa?E=;;j{a
zv+0``b!gSZka4YMtnF|9-Z!y<PAIeLmhQL9br(ip7qUH8m)%1zqpL7@usCjv6;&h-5hf69A9OSDN@!jn5+)@qmr@?ilLagCfHM7CIHC+^_y<_Hn{61UC%H{
@Enr6;|cO=a9(@0ezn>RySMeed?>t#4cim;2h;3PIyr6qDoP!pZ#F`Yx0qqXdf-*+5nIF@o%uNYPz77}j)=*pgg|L%viVo6GWtr)*1t~pLhp&egIAp~*zd3<
UBx2@)gO;6uJD;)Px1Gh2Rl}{`2s2H+~#=L_f1w$fd}J#d*<&eTcAZwDOmoX7ex1qTv+^IgEK+7#1bP0h^@9JwYJrOqF+Y90}6Gly#M9M0aYj$J5t{v%)xz!
RPxM#6S&PdtkCi)1na{NWt=Wr@SF7TZN8C$fIZY{E1&Jae10<to)>XIX6P%^-ej1?x0Al)io_Q_zn~~XXRzJTTMhbciBFXcmDl(!prAzi)8z9xklr$15fGMz
(Sz=w*rSff<D~1NO3j0C)cJ2Ty$K-VY;AjS-x{M48&&oth8Sv6PO*2egElAqSL$qQTs%l=#k6XI*Khh1$4nIPTQ`g=JWLysQ}L&EWF%4G?MCHyL>J6aAA3*R
t0RrPw3vN58atF<m3R0=Fg=jQ#XRhQ?5|d>dF#?h&#`CNDl&rV*l3cQtRv!IO8owpOYnV%$^Epe#xT`BeqO>+6VFL6>|_#h$C6F{6<?e-jBnl>vDpg8p2xAY
U2Ee1<@ceAue<M@JTcY&t4qD+I#>?>VdDI%3!7Mvx`F^pcxB4z^q$eh-iK3{zkjpA@YCxTWzroWSs$@2(&~jT{z}<x$6_GxiL=@z)*Pu3MX0a{g1S=E5jw3j
JjpVwuKnTwt2;fM9Mqvmu)dJZ-x-FlqXNFXH~dh1_Va?{mq?hIz)vf*2{IN*o*4p;==#MUxE0ZXa;lI7iPvhlUU7`0ZYv+Q@sH~Ri!Wh0&!<L|HXg=moDc5T
+k-?W`raCSAY7J*V;8b~5dYh+zFE%<p?jr1F%#yH`>xrC)J7HPtLOCVHk)C)Z^8PwQ6A2pI4YHuZi|Y*(A(cF_0e+2hJJc28kK2pwS%u2VWCszb;=Dxlsuf_
VEm|q)1eM6`!hvBB2DA?BH0=F{!MS31Vcb0l%KR?uaCVX*BBU0Y+y?HVq43F;ImgE%K6&;;C@B?bnt=}))*dM+$&)U%6UJYOBDR*z8)Anb>9Pq2j5z!E~{ck
PxlF3Lk9Xs^OPDp0x>Wa(ZDYef=S(mfNypNAjx^Tue&G+5cxI6e$@|iChcV5h4G*r`9W^tau1eN!3;r%lQDb3>Ga))9>|(0vWxm`3`(7ilESbIZ18Yc+>nfc
2`}BEm`@@O7|E?PwVHw<>-Y~vOKbd!o*EbHGRMS&L)U+@#-irg(jJ#wLmYgwE%4no8}*0ipV!r@gT0z*;RAthpMR*gJ(}wW67qxmKC4<_Xt&<Kmn<6Zq9RzD
UfANd^z;Q*J3r)wNm@IK<e*7ZzPpmi3#vW=%m;Vf&@B+$FhuZ+E8j`vK0MWd$r&*-fgjcgs~O+VHsOzfhCP4fT%F(*SNLYcI2q-=^{b>mJfQvWWY`~DZz!4#
k~?14g?Gw~p-rC)TC5iD*Sq?FKP9ixU0of6N3T$lE$YFqhta+Lq71U{9Hy}kw+A)V?r)+`?kFv)=!pE|40+xoxw5Na&~IBl&T=Od(dTFT;=1gS{zdK`*@hW>
)(Us~q?!>xoE-3xA_FYiXGX{N*kMN}z237j0T*Qz8<!4;;@m&(ILSR)$nx14J#a4*%6Vx8LH^e05biSg^SB5vG7o)AX0?H{yeIWgLK%{;T)lkmw+8}0tPo`r
_-6B^(-(6q4~!cY?()!u{I{pQ^&aMuy{<Suaer>U!3~-*=No7rJL5a+$?v0)CXk6t(fTcFfOolG|KimA@zz$P>fBs1p8ISdhB*d5qAQN~*aRYKOiWu-*b7U<
v++tfo=|ofTo{s2h6QiV-zq}B*R+0dqPMNW%O={DOhI3WMqT_Sk*EyCEwWeS;Xb%3!5ucmp^b?p(af|T(HNj}ee<X65^Be8iN8;Zz^VE#2Y)_@MtGg1nIj?h
gH~*wolMDq-t@Y6sZk6(+m~PTSE@sJpM2Fui34QQDwdeeMxm9f(3{jN21GlxWT~_faGqU%*3|8W!H8QYzr6GT<DEn{?cq37*S_;J4o!ztC5u?4bP^P+Hm<BM
#lT)z^Uy6~U+h<toMMu8fDrW=g$HrYAXVeKHR9+9nok!KU)p#<>yt!(M?xbKuEd{wo>Kt!)11_Lb%RK!^uKv4G#|Qcs>(;yv#=;a?oKY2gPJ@3bYB+yAT!f6
Rwf$;O3N*`+^#G%aF)Jir?tW#BlVKVqnW6GJ$ox@HwSmcg<saQ`a^&5pu3t?9ZoIs2*?q7g`8E^;v07iqGY|OUhq30^X`$u=0*WXOK|x6U^D^Vw?D@oEh<8Q
VRYI@DPO$*9sld?_asDc3g`+6%OK$2PN(8~Z$#2?d>rv`#iPn@(S$@h+}UayS}ZidLsyE%e+Qg#$!xav#vOBH-D)f{S1-iIndqmgHhCbnVLIU&m<PYZT6T<i
*|<95AX1l-2D@EHck$Lxl;12PF7Js3zy86{eyU=)z3nTKoXEp9^K+I0b#br?6wjtM%*M~Gpd_6fW2F1GRHWQB#~ll9>KOwgTxa-Q=4GD(vbGHS6D5K8Y&dg#
Jh&7Wt{LpCmW|_dtyV%)o;KP8-x58pkAv9N`sed>PFVR??9#0njExUeIB{Mc5BTUvx<4eq{2x)p@!2r^mfJgai$WhoQrs%n{Y25PaH%BHBnjmEb`^DE2Jw}0
V$-W42LBr9T87^xVk=oJ<<pBm_<bGk&^I!|OIhiWxeZ^uwc<>0I2-}h{Vdx;1LpW`qoMvsCJ@Pj%T4bt7(*&axJ-^W9$VUXKa6gc;CW8&qlrK9FqvkVc>4Pi
EFK)Fl3~1s4i2llvvP6B8x%a^z#5IN`yF==gqh*~$t!hqWX_<jyrt8~7LDF%=2*tOAY6TX#9*&@3P@=Gs)RVjBX2C=0=JMpZqV)7VZIoSV5^f`2F{s?W;r-Y
6wwBa>D@a@0cX(pn|w%^xByETuh@jAOfYz8-B$jT1;~f;w+^L+BFw&9hf$>xPXD4W?FRV3NGG0A_>MiE&>O6OU&+9HC!7A^abK{qQoV9@alp{vxy6t|H^enF
atuobA%p*7?5kvJOfL+-Ez{|Nr?Y6+!NfSw?@JPuSri4+!-sUg#|;SHrDdJ$R)<cC_IX9$W+?TKoH_S98lP-eB<nn-k(hsGw$;D}MP2Bejj@6Dn$V4d0)m){
6J~WTCV1SH?a{CvFYs}2mc7Zk3`2jg#ZwSGx!?q~qiGSCcq1%CKd9p8HDetLqC!~k74YSD7URM?3E$qI$MKWAXvDw!GF0cI*plo0F#Y~YSj!20WY3ixaSe}$
dBss#rz$0|PuGnS&zQksU$oUX$q<yJv&w3)dtg-Q24(FOVSeNlmk+Nr!%~z!_ltHO+T0^fXR8oCKk}eWuemr}6N4D9{f)(saZ4@E%LE>?cD=oFB^T}fPc<KC
+knO-^mJI2Eh0u1LzGM`Ax)#DrQ6~QOJR~PPiM#Bw_icLGu#?fWfQaON_P0TQD*6I%;3NL&Qm7)RTXp{B7FN-RBKGYLd1U7qFWbf3Prb8hKx}eZ%TUUOazuB
WDlC>+Q2B{eQ5ez5e`jq($C06LaJ~`q<^vi>AWdd2VK%Itw~aQ>Z2QI{OR|<BXWV(*@L|E4kmaco$5H|oPsqS-3^u)dkh9?sCd4!g!|=p%S}-}XnQndXZY3v
9Q)$t(%x!;`Ms_~g<T+eE|Ij1O%>xJ15Z5lpaUZBzI`)7uZ%A$r%Ae<bZ|sOd{ufe1UYlkBJ5o8xQDr&PtPnN&aSw6VZR<S=5*ahbfl5=F|qBkog@PH-E9&o
N`&*<<4kweZBX}mjIv8O17#m4Vjg?C!c1~LXguBq6l-fG@;ncq!L;R+v`gr>i<Z}aOod^uR)R+UO#^(r<IEL)JsuGSTVK=3Q_&!2?rbO;i%N!uO3fWbh+kHD
u$d)+jPYf?y3#6~yw9rSpm`a$9#{B!eM-g{Ymh%#s2K!voo2!{W3ePTcbL@19?M30pI()^fL3+>-}O(S5Ky__s<7w}uN%VwYt;gv(<;=xx8(_5rEdl2P7=IH
A>WVdq#WMS_%8QK^5bJ-_%PKkThzvWzGSBmjki_G%QxjsLAs!2`JKW6YfLkPwCm0YOuKgOrm;Kd`1aGs4A|hdarg!MkUSKND%QTd?+2w69j(#bP`Fi&9m?Mu
1m8KL+y#A0IL;<1Th|1G^0?NOUJFzB>`oo!(Qv`*q0#o6iTSw0)ZO90t%EGd&Rd<SzTo&1bw*6C4$F_X_e5p8V9SX1G~F#<cqSwfXFg&^y(C9Swr3*loy_AB
4h=`a`%=51NMBSCU8odllg2la&G5}wZ@eRBYvgWJMc<m*%7C>ycE4<}W`#Q9@@CTDwHra8$?SS3>>Li2Ddpx1dp+^s$vv%V0@ssAzcV6#>xF?rl7WMMi7<Gz
?y}E197DS)O3s#UsFamRc$jX34>InVxmUc<t`x~A!WaT0-xI&Ak%^0kd3IQJO+isgpocAc*!p(;lv1-m|IVK2)izV4`tRBlm<B?0!R#<quqj6Go@(hl>4J6<
l6RfX5lE+1(<DvI!TjSVYw64(IKp|P&{8uAdZboUzMoCNcSCgP&XZ!4IxM9UXS!lGz@?D+Mi}z1QXkcS<^JEEjygLKYmPa?%kh|V@Vz)3kjpiYD)R@stNSR`
o_wq&)k{aU*kIqAYu-cM(WsFTW0rRG0N+s6pTmsapu1^Ye(Fyau1$=kSQ-YR^Y!+-?@l(@Gx~TsQQHtR9g)=~N=Eq6-E{HTfdmv$4ZiL$@<o81-h}X{ChS$!
e7dA*4>z|~+RRaR=%!U)<f#+~kDZ&Zj9?Hb=%oU`jyj@U!+Tdq*9WvU!`083Eg|-q{f5&Gd9+<SxJ~|37)8$KB+HhZF{6L!X#0#RT4VVu%+L9NDPo`0!vGuD
n8XD4QJSG|^DXo3b9y-VtM!fcgfXIssOsD)wea`v%;X6wQ&=mDn*8I`fqqO$s?S?FLZ30e_0-dVmt1`0jYkAOopgA~_(&h87SbG6$OEuPkKN2iN)7hn(I;b#
JOF;Tfh)<npncT#h4hXzMqcRKd^FL)u~pv|xBc2scCCJ#f6f$-TGFDvdniGpXyU9{l?lAg$b6d?PzBQpeb{AfCm2Xyp*@^wiH9FZPNvP+qeo&SY+sKaJoX+j
M5Q4dX^x)hBXE_FQ~VR}Xcep)=!b57HAcCr<X_hYT1YnxUP)Hd#`Y!=U5uq8+^21ST_vGMN`~3>Ky7WzdknYH+9}|1l#Sg5Qg8hG!Sq6bSqH5y8oeU@y5Jxe
Jyu;MjZ|X6r#r1ysBRq{GwP87+3>p8I+G4YWBVDX2)-Y~cjEPfSw*mf6g~J|V2W$bulWnE=^)qcMFah1cEmF8Du19;h4~&faRovjTdI1Iw0P^nm6+DQ=z<O;
qe5S%yXhhIe9gk4(>~ZDC)X(Au*Uj^5rcGZ49fFY{SFd5UtlUA?!ha4jIj3?z2MPB3rm+yi>(DD)|HzWopsT0r!4cjh6%nIs>Wm`>cQ12ac{Q16E^jxC-Pb^
VkTazy=cl9R;l5_288cvnxB^2SmTE860dC>=VYM0VQ=%m$R3s?2lgJdP(!v;r(}h*7D(S5$uHe9f$CV#nV=>GTz29yG#XXMzq)%*)!tj8V1R+?%#;}xpH!{0
zO+N*q|9(ist$CQcuul9XreZYlRH9K566Bee%^b|1*P`P7l*86P~BMglgZQtiw6=l9!Klp(Eanqx%u1}rFWET8TEh;NALQ)iZ4j@(;A)%C?i2j>Q-2yGiuby
KA(PLf={8lA)-eU;H+9SA=D@jAF{*&siU@NmyLDdZMH#}-{Ph8K0X)~To!+P*Bb77Q%_SK(}!Em=g;+&@(A&HP*%iVfI6;!zwWR)fyn#Ys;{gUykD1`i*8W>
ooSMfz@#AF#NW7CpYH_E8yi1|qpb1lgRDh~xep4Gn-+0h9yFw^tNSl{!|>(d&v8w*I7J@!w!1YN(%%w@8D!+anSCbG!_){L<7j0ahZqqdc)&7juLCSZ?<Isz
Tf*n}*|l3;=P}6Br+(eZ3Nl(Bd2R`sfco=-oArHTq>U%dp*Rxp3KWzqr>)Snb@c8ZF(1ggnf{UrGDGuE`e(L`ws_f`q5RCk2U0DR>Nj)sASJDnXsl$7TdTn*
<w}GxG8j{6pY8~oMB~~L9(I^W6a`_)1UAR1K7XS11Nq_GNxh6-cp7?4^(nJEh#yG#p4evxk*=!UWdaX11S{TmzpVtimA!?pvc(`LP<}+`iW+pvM|1lw$w0jS
guC;)A#NpEw0w3AM?P6koY8<S&fd{5;Qtc^#XWn5*a&;;nRvW&Qpgl}YiG|ZaoV7<l1Ssnjz2zHZ~45~Zx14K$_G^4vY_jHP2W3|3Wi;F%h{*)D7^9O<)fI}
;QE<;Kx^`U|LY)La7S24H4;@RB9$wGb~u|gbh_Tr27&VXeGN6V@jG<?cV!A$v<^i4<9h3WQ-7j^`CQFVLY?tT!buVG!fWmJ0rgPnF0f}Sb4TSy<GO-yJHANn
Cw<8&2l9@>+glPAc=Mh0qoZRA(y|-*CONL4;mFg>G}(A)E9j?A*r$T))Ax<G59avxym>CN!yCPy=-DM>bfLHIuJ7uu3%Lk7s>T;4c-uYItn*h3=WiLxq(2vg
frFj%uD1i)FR3>iT+D>0i}?ADzxt5y8Wwkwxd`3^H54h*`S3U{<K#U`35g=zJYo@lY)<UinoJ3W3ie4JF%3Zf$4<+)dsDI5Oy{9ju7l*T`GGjr1V|q4K2);m
4ej8>G?VjnXpkj2zxd=n{x0hLDR)apVWi#p-dH>A|4eepJ6{X=+y23iB#j|j8=35}Ad8V_NmJi`>44a1VcYR<Da@q;UjLQUL(ONpEAKlkaZE<DO)Db_4C2q4
imo={8SN!mQx_}D1nTK0_eVmFtbet~Odn0v61CK1{3zeyCp&+v9J0!C?<7tcK}9M}@TrbCBIXX7r@v7IrxZOsznwm)$H%sbXPYpR|8ZvMj5}(>q=ia<N`Z@{
ww~dsE^I4%?ugL3LcZ)?-`Xu{{GPo~uSuqahWW?UBnNE3^*2q{$txIYNmA2{)X|V1H_kJCY6yo&|F(>hENCAh<2)?>3T+ny`Hz2g#tRCXFnS7GP*P;oxqsD1
fYzv;yQ@E;*ADEt-|dTo7D-z#j`?C$^`+y7AQyZ+H<O$uyC1%V<{mUThKSzBO8L;ifUuv3?+)A9<BJdlS64F&3Z2YkCiQ%vU#)5Ph4KIIQ+XH#9dL&}NxM;(
k^s)#VhAHAwLmVTBlWSzUWh1+tD5mD!)P1%^Q;dta5omE+NY|6C8M#00~4aK>->@Yw>%Fr^daVBKMHX6TuaUwE<&De?{CY$oC=-2iJ!<SOK|=7J$`4R6il#x
FRbK|$M{yn4~9r@<ZW8_s_(Nz0*4e$f^Q-IJUK0RH6R$Yi-n}aVoG3Az7#LDn~R)GCts!m=_v9sSTJUCLCJG?rzrT~gZEp1Vj*)l8U*^Oc*-I4LE!}pYA^J9
$$Zey@WDeH_LF*v*^r9APN!&R^xvM&6(`<KPmP4$+vm@_n9|`SoOm}l#2PD;A{q0g=6GM4btgE|1BdG`I1al6;B(-dpYcQyibzIk&4UwhwRwnx$Sx1G>N<7I
OZuqfW_nchAqa~N8rzL(x{xyH`}c(_7LTjjm5PhJka6p8Soj+Quz!4Uo9>)C%*7uJNEv0|beV%vR7WJ<FX>9ZURQ)u^PJZ;w`fpL7M;-YA^0!n!rts2!kk=M
PtyL)hmJ=q<p*5Nu_s!@V9HAeXTuAA?br^%!X!*NeMbk=f3iEe^QG~uDKNrw*$Sb5BJapvvO<}wSupuu3CPS-emXvH0`;cCpwMkayn9K>^fy!+Pwf+_&*e$t
HN))<(oPYSMRIa}Kcxod>GI=}wYp%}%s61mYKWJ1%|@+qMo?vO8;VV~!F6Wx2$C{&tSp&P|C1BK&TUCfe@j-}E)|(s7*xi=c(r0rRas>0+&pM_OAn*}cDgQ7
`G9#`U3q?31bwe#Wd~kffU0;~F7r!8ybo4p)|EAaGSeGkv3y-9O}NgnelrD6zJ+YlUM1X^+;)6ME{m)_T<AYW@YGCpO22yw;3<#V+O|Ift$)-X4jr(=c!s`@
*@7}CJEPx!ncGY7r^VeJYH^sKS?{umm4VO+F73<40eGx*(j<?-2cId1K4_Xnqn`HJrzl?!wB@srJYe(2-r)=h>&roa`{y^i-7ydm9$AUqw8ak=)&~!hq|kWy
2yH=`3_Js(k|bXFAm#l$(>#w9#3i0jim=NeqH(KKCsYq-I_4<8ZONfQO#j&N1}%)TNS*npr;pQ%fgv@P`Y7?Vk#+s640+e`*L(W(z+GJFT#)F2N2!dLqs*-l
JAYzQu)q^u#94KokVCFnR=LSR1*pt!n0K`pW1aZ2NnDOCyib3*dDvMG<c7;F{2QhaeQP@-dRqlru^ept%_0~wh!g$RZilrDoq2ZVY{WfJYqj@vK+!@jS?Yu=
+?&P^O~jhup>o;o4yzJ~Xr4@b>e4}_Ih$ADK?_7hUwj==X9np5C%B&$Ye7v*olffWIc&rSYu|bAiW~Yg<|EIw5m<PQXa7|tD0~q)#V}(5wYjHT%gM~xUG|BZ
KVpGKPY;>J3LaR+sw;i-A;Wmrl*@G`GmNTdzRhxQfg#VI%~PZbpxM8q*kLJ&9v0eoN-k}z=1F{X;7)``iK}-4xiU(QJ`(ZU=Z_qooL>=}A-H${S%7|?JQ7d0
YZzxpW30_lf2oxZ926mA=CtRacj_&J&XhjhG>A!1uNb3F{yObtH9MpgBx_RKG6eTtac+BMHTX<BwHL}>gt21K$n=^tEN}VKyZunZ(S-{(_5@$mTjHiM{cDLv
!>RM-T4x}TGjq5=wiJwXEOGAD#$Y)nU!CCYfhbm9%EyJ2n2Gqqay~>JKi`ftoKBD><le~Qo%jm`PU<sm?(+i!9n(PDEg>8mV$LW3AOxz)pUIlfozQubLY@4+
3ATy`noLOVgGn-x>Fr%rtlYPfZ2oDF@YwxY6;q~|HeF+8AlJs)b*|~z!~S@WkaD>h708|`uIo@Y03V+`{jmKZY#*a1P1;mOZ$tRLvMyQpCKa<Eo!5k#SfrWV
DJ}SMt4onqh=O;yeOqZT2iZY=fnw!Gcpfu#s=|O7>R04xo9-%O$nEV{Co@r`_?<uW=7%Se<22Z>(3&H#<rDD=sWPgY!fyumDPv08Zpl;B50Ye6dV3i7(K~OQ
EBrzmWk<XlpM0u7pg7-}^L0a<SZ#4gxnqu*O{JZ4b&k00tUdWB-U4(URlM~LI!HXB`R)GZ&VSFrYb=%-F&cS6{(0T8jW9_Z_m$$iJ0yxFjU_MI+m<LPWECTy
bp+={k>8W#N{DT|a)_2$9c0l*&Kf>cz>;43@XH5Lctl1TpnTjCO0_yyFEGVn_-ST`+_n;`AGG}0mUn`>MmdYw^UFx>yKu4jyFFCiWu=cE48m#h<w_T`a!kJ1
Dw~Ulfb+Zfp+*{OTw7z${QB1wtnaB#r5PFFEwkpNtCAZoRs=<g3)q9|$1N(ldIOwp6nt<u)e9^8Rj+#1IU`Q?rSE|#Jy6%AatZBTz#H>lvWT|<k?p(ZSI_gJ
XYzE-*|V<j-@fSAXLku1ZBhp=r$s`{W;QCfxCr4=jgnO-?Xa&s-srodCu}DLgkCZ^BG$y1_tLuv2p_WW2w$~Al+H@1tdt{o1#^pc{+YopdC!DEyD@S%>XRxC
>4LSG_j|s)ELJ~oDO)(`LG!1cY0GXj+9dsF$6gqqab!8m-o+NrkBrvL`A6f><Y~jR4EiuTKI3G0I~XCJM0^xden`qX&*?$ncKU9CclM;`AZnU-``XnkbhFfL
d=GcTaB21>c`gGa$pu&mFR;OGW#PHB2w{$uey8>Anxjkca&Ot96_yfmMQPg&q4B}Gs+7Pp`Gs5*dG|cvS7+uZ)~$@o<W$PkyE=HXET(i#+6p0F>V2x8(;%_r
Q&4Uw12Y$oHmyCbczmSTR;oB1d6w0eVj2pOzi)V?_=_Db6tuNuu;_!+JLY5Upe@X{76hnxq7m~`!ss`-Glun)H3L?Ypy4+DfkiR~2K0{(8(!wYs<+hHO-^GR
P#~gIl`(+j`^_6<!9gIowjme8#DSRZ#k-uX-iYB?cjCz~!u7sC-|ii@ffCb1q3*mI$iDGee=swIG_Td5K%^Iz0>v)%GUY?dI9+G>o*ImZO2sdR8^HR<zn>Qq
q;Vc)+j`OQxc&H+Z{Mp-Fp`f4Y1ygc+X4I8clDPLZS<(q^LhcYx=n6|lq5sm$8Ph}Pce-6?=iUS;)aJDx3u_8jM2j&^!ZeXESBp$mRJe%nlvRv-`ymKfnN^~
9JaNGz7}KdsgOcwOx%s`A?(k+93nc$h;R^Ry%ka;bB13cZGtSNHTF+?)fL=uLe>KPh}l*-)R&cXnN1VG?x=jWrO*36&e}Yk@jKyACG5Cw>DzAPBdaa!QLU9F
POf`QGEtdAnwUhbA={R4-oKoEzd1wg?<cd2WA&KG+H>#C^)x8|`D37Syb`rK#Ew-XUTBy-5j>I+iT#!r`)@y#LZs6pAB{j4@I@C-3YrJwVUtRKcA5(;JIL=-
Qd!`a;_t4GzfB;a_G5qPo()6GTVE`K6p@)s>^d@k3GLN)_Cy6(gZPJ}(pyS1{Lno^MI|Ew+0%LqTK0Y@YhZituOI_WqSK##*oNUJ(c^80!!D={uQ++*hc+bp
Xw$R#^%0`d)3Ma)3ir%c5>G<qF?IBwnj*Cwdidv=YbXdjD0yC8gVGU4rT7*`7BoQ_w^aVJLIbp{mVK-0rpTOn<h_qq6gkGyP7W1jV7J&X&3ovAfqiuJW4|rH
C|$zG-{Oow?nL6bGdgfVwG$&VA8Pu>+ux_F;YHKP4+cpkkewy%>ua%r#pdy!DXMPpakhK;U{n!?#zF$5uasavoY1)DXo6E$JT18?wLwjKmyV6o6C5ng-#j(^
VCd5zdLqmW(T;nj79Z=P!0M1g8l^orpDf;*_E5pL*3|FQ5!^WB(W_j(YKeF6$DR}yIU{x_b7JfUGZ?SVGW)M<qjXv8*>a{BA`b<`Hhf5er?^$;Uu7@kzAN92
9P&iZRkyY;{JQ9WXRR2?W{XD>wQp(}oMFUEI<jTs4UyZ|^qywBLfm$M+lAoyPNKZ&hN>2z4*kYJP3(k!<N7h=4i|Az?(cW+5p8@IQak498v?fGSb>M8`Y>8&
3?C`60IvW?$=>!TB)W<?mo}InRMxpMCPfSF<}YuYjZwj|!2CmZtD@0SbJqM^ln2h;(Bj|qv`4{_zdT>bjX{fl@5J~e@L{&OuGiWIRf6WcgM?hyd|9x+NZJ5e
+AXE&mMRDi7S4Qq*AdtMZSmLb(Se#&*q_od8zgd@maTrbL<}kOSVxHyG&nh5e>m=j@dPy@3w0|D<WLWsP_&2Z_{GV~eQqF9sS+cWGK1`)JBCH9Hel!2SvM3X
TrZc}-r__#xKC?df0Sr|isXGc<yUkuEKdE$UBC{s3My>2H2%0&_Vp;MloF(qByV#x`oT>nWQIpl29W|xLbvk>oRUeRo~>XFXClkE+F%F7|9QG+Y1IV$7||oK
Q-Uc?uY7xe9b#uzMAYA11(ia3N6?@N8deO#UmAtO{aokW@&zY+&H6j3rmu}DeZL(VWp6}}FTYaA<3`256jkR0JFsQFA<r$=K|@1Cyg;EBrhB&6dv7>G^a1tb
AJdLtmvX<lr6GsDs;452))Fx6Q?NfjXN9LK4gPoabir>oAoB5@Hq=<MoOdd8FmcEJ;x#dG+zrVzo!+X$S?OW2sz=7)V-fWGCGUj;9XSsIUvMBf@#y51r8aV@
H)0a&?Z7oRA4}V)47oaumNzqgSh{_W^BzABUWS_#FpP?VTi$hURWb^znj!bx?kXW)rakG&P9Tn+2wFeAVSyxv@7-1d!7x}#j6R*M0WJ>4W4y8izTx0YtO|(6
i1P=Fmp3&~nK++*%1aL?3P=;WL-*kPL(}YVSv$mZ5RL1$nZa*AN7p?gL&W=B^RMc(0oS9A-q3JSoV{~EqD#yJWWCJaSLiJ<?QOFb=`DmD@_PzZPTKg`x?Iia
?TY7$swVW2W;kTrd7ITH1OlAgvJy|EQPnSKLT#vv1VeIeKQnvO3?5w{mX*hTg~Xb!W4hpYvz*p=*ooi?38{6eYOo3(a35W@z+s&XfseyZI4SQg9C%$9((4kE
Z!&e@_H=Y-HH`7U{2tMdS5205hQ#Y@|2AZFaKrI_Rx_OpZVxy6dGUvVa%|&ng0~v<GuB&0vsI9o`ralg+ZrT)OA<_%w9$G^y7j$JE&3B2o7|2WV9G&>QLEY;
f_Y<4Jt`7l8LOyNG!_D>+S1$9ud{J_?&;A+cp;ETZeN+P4|GBj(>p$ffMu=vz7VfH-f_#5hvWrf{HgJXyiO)w@7v0%N_K}%BjY<08W$Aa)TYt<NAR~yLHYYX
T|hJa%US<&JYML%@a0H|#yF#|qlU9LROUB4|H=BG>O0fDs~2>UT(CP-SRoHhe$sXFPuVb0vL7*Jb4B<FPnS(4fwRc>KjK>ShY5qZp4%royg7L%?wD^5@`F}a
_V&6V=|mO#cCsTb%ia*iacvyZ<_(td^@5K_(NQ(+T=XSu|JFTLg}q+`9lf6Wp=K=eDh0U}Y-|FEe%*+McC<nzJ98>d#5B#_s4l=aS(UFPewN^&QX*xWiwDk}
FIrX(!l|Pdx(*Qdq`1J!`|R;V*lS$7SEn0=$7cNt&F6)Yx!}uI<nM{_b6YpgggB$0x6G9&%>+>$i8qHo$$^CON`t0GII>)(=Ze?D5plU9vRB^_zSZevns-tl
cHcPswq7#mi)qWqR{dc;Pdiq9ED%jI`)pZ=a&Y$D)0wAL{x~IbJ2LWl9M(P>zVkj639nNY&4Fwo$W{{l7j-Hd&4qs>Jr{Ivmc7QQL^A+f&RJJKOqSqUwJq0-
sTtB4Im5_~dtiU?p(B5NW8rCiRFS$g3Kd5AzT_oIh&eaIIFNo3`^zl(So8WY)>plKEz}ohruVCiz0AX@4mnl&EJY-)F_$f#3WZANneDAgcO0`+{lYow17UEO
o$L2TqHn<`p7TKn`DMFmQ0R?;mAnXXp-{YAR~xeJ$wPQUr!4!g6lj{v9|~`e!{W+zX4J=MLay8@U47^cx|F-E!X>u2<>~#(a3BB~q7ywrmH{}UW!A-SlZ~|T
=_9d9jtGm)653dEMb{^jCo5&HxcM3jrFFjWB^Fm@Oba9UJTsTAIAMM_oq6^+JKz|nro?5d7@VA-vl`#XM15ngT?%^=0@IJ05g*RO;&ESk3hDs-G;uJ$7!-&Z
dyCyAZZAZPA89aT4~Ls#Xoc_d|J^qfOJll~OyP};#(QKnYB4abJ^7nABN`FNy2D_aj(3r%v*-K_p<mt4^Q=D>d+r~zm}Ip>@7=s39pN#!Y_6m@SYHW7$LC_=
N<|Qip8m+|;Rok<t5Ss!W4ztCYGJ)$1fklcvr<RWaeAIbc!6*pR|MZ#n-jQm&)Qg5&Y~y&79??K_4>oYyzE-WKRL+rrod7o8Y|v6$hC;W@sHuY)^W!q+`91W
6`Ow;)B`W^x30RwAbVX}bif7?Bjf&WZd+q<)6phefWT8cbVFM+F&JZU=SZ@QN9xeYDSQ58oDq*XwAA5;)cl_86S;)`{%gSUqRtc|{7+ST2z#g=Pcln#&lbiL
|NK;cyCX|@)<&Zt4)Z%F29G+~AgTTDOaI{jj51w+K6vsHu6rF3@F#PC*&Nkjp^YG1%QgBkzw8e6fjcK(nA+o>uO+oQc?4Wth!;*#I$%^#mm^0w7QNw_u1B(L
!P5Dq;A?9MvPY-Bk1={<-$=z?v5#p`Qx<$=t{0DU?-qqk`=h}>LUyDj*acD_4-S(>Bx2c8mb$w?8-?95_sSalvCJM8k#;8(ukIBGy2J`YvUu_GB_&@hAND-i
9TJT*_s@DYwu@l%;eom5yG{rQq1Xwuw#K%trM+gGA2Ofukgv#kf=0q*CG3nd#^Y&ro)Ng|0w={xwO}~B_Jq0qlQ6+XxsuEAbSIq2DQPfFh(PS2qTZSTJ9y=*
8-z(kA#LJHlfbAAt~7t4aU0M?evJ2H;t!TsbGr7nr6n3{dAbc3e0<=|ZCu#XZjU8rJqx`MI|x0CQU1=FgRtul=Jh*^P|T~g6487cYX^gsOwWYk^gqYm7lm;s
v}m$@bk+v`)X_X!fuYFz8~1)IItFf+JUe@bz0h-}+)yvTA0-xt@2MTk#yXWtM9OhfaFW%Ja+O4YJI0@`@tZ$#+1t$zyTrg#rR?;at1hNS49gkVB2lSN9^I|%
fX}8w3LOvJkY=)Y@N`!mcwe4RC{(cnMSx!Xtwl#zdip(UG>O7-=APFB+dddsNQpM6H^!{Ti_uf_F0goe;p>@a0T^+wI(TEhBaWSPpH0#9!BI)E=kBz+@bs?i
N*0TT-1(mK-T#d6XszG?55X6Xx;{-DEQuj_!?yyO8!<@dII!}fh2Z1vk4F-I#X>IpZzZW;H1d_iRj#RrqjZvmrk2wbslh7p)d%hI@A7DjrJ4<tPet939MeI)
`c*xqdPgiNz4o~LR~=&2WMnq$)+mn}><c37`!u60zsza^8b4NTdC(`ruxk03c)t~t&P=KA|EP-AZzcKX*dkym@J-{wk|W9$1&#7L2zy`a@2SIQikoEi=Z8JQ
@h7BeU&yd0gdZO`%jc7Tq2ReqryMJAYEyL(dw4-}I6yGw2|o-}r5u!E!azgpzur4S;MvvP%`gE$_|^!oJ)Q~1k-TC5*Hi|WI3G<_eVs6$ubRX7DkIP;WyiSO
u7$Ch!@4F;ZkU%0Dy#eu2%daTlQ%6%SZ$CIOInY?AVpjb%^Mr6%MFi*bhskzqwB4#HWN^6q%iD?#lzd4d{XcVFRq2RU61xlMoi0oj?l?yY&Z3DTTn*hc*wfs
y@EJwaU2c3v~7ql&2^1&yl$YS)4k9v?1D5$o9um&4)AHb>1|Wzj`~+)QV&)1u+MTK#QKOCLdrX*GkFssyq2DluHy&S>dUO5Z4r1y_v+s`rwLXnrh2{9J@KIC
*{@hrKZH#UNFE7`fm89x<YVqWkO@$!xl`f_%~J!K`|K^T(0x^_FEJ5DRMlr4YI0$_A=!DMQsBS*zC71@hJhg#Y&u6tYAl`cvTbhu(7{l6`|PQv+7pfG=ffSr
=N%9+X#aT6NH(H2LXK~WIzs;Be(Lzs24Foc$!*3Fg7~)H#NnTb*sE!v)%-C5(>489q>jZvO1oHkOeGpuZNt6J_5{M}#NCpv3PO*{9}GI*ors$P+_on}Q!(3@
6UJZ^1(IiSWO>Bdu)5(LHO%V^R^iI$hH5F0kq~|-&k+p8^bq$A-fT#+pL=#g+8e2#sZyt&1i?8>_wKf46&ebp>)s2*K&Ww6Un#%_`^4XUFg#?6lKf+NA|<bI
P1mq-hVZ^s8q}-*4CC>`cI}><LOhsD&B<S`+Ji`9$;MzO1Hr^M-OiW$qW+Mo5Rp$9qW_&%d^F{TT#Av$bcWineiayXO~D)AX!Z(gYzM;au<?WM9?AIPIsC=r
_avh9s|GXo>LBp^N}+sB9P;&TfAYF|qIT5FT8cRyL~7r(<Sr|NGoxX!yeSUsbIT*Q{{>@U-q^DLn>U^dmE5(L%Y|X~i!b(r)u0QAI$#_kg~5Vo*PGsH*uxkp
KQL*B5u%0PaGgeZvuZ__w=B}`b-qb02#0QEt`ECvF&=!CJ6ZI?5NcHQS5~+lBDR+N(Rq$1xY-m=xk>naZJx66n%7(hXZe&@y<8%MXryFQ1p~oMe={$zJ{Bor
T1BS2VOZ`>DVGgShn|*TlcJUz1o2Wgm5}EaXP?r!dj#U<TlpC8@od~V>M8X4jxX}xES6u43dTYEF^X~Oc%+B&#ni2P!_<j|;+9A@bX(d;RFZA6_~Wzgx>_zS
j5Ap;xK^Th)Gzw89KnlJoWCxy>SN=JaU}<LJj6Gv(`gBu_0X%CrnJKo?{*$Z^STjuMBi06<9#Axo8S6eqK*LTX7P<H^6?PjD)IZH;0A#h<LwWxBQY`M*O>Mq
9uLG?R3<w@khT+`NfP7%nuP_k12u#>|7AlvYZ!t>yQ_C<sDcnbf429nK`2;6ejibttHF-xaZ-aT?pQcW!@$q(h0%{&0?U)Gm^(<Rc-J=pSB>+_9)w2V_M462
VCi7k+lSsAp$LX~VDifonX%|>I6+xel8pu3EsN%7;V|oudgXc}8%M`asXWt4MT%GV!4G||@L{jza-)kuUJtWQBd^bYd-_Bxw>GjU2!3&{H>ai(kjyKd_V-f=
Zk_xotR9$&XB79@9O*o;n%cyyZ<&i=hUTgpT6U<dI_=`69tib?_%5IC0ob*RZ^|%B2PUPNqcVKp9QtjyIl&s&0y0_JKL(+CPt7u0eHc6!TxJrML%{sX|GICM
E2w=gUOgX}h`)xOv@46jxJmoP)2K}w3+G?j_g{6#-UUCs>%M7l+`hm<eLo+2OW(LmF!|zodRh8|a|C{McAY;556l%lTP%HO2P?`lu2BL9t(H9g@q#E8+*hN?
GRbQpA$TJDtXeS=p32ZjJV}8u>6b^{Dlza1qT5?R*td4WM{_RyDd2d&KaA^81imU+ZtBwqLFTWdjD;=1FV4wI+a|h$lO_Gn_f22Wr)Rr9)Qtf<X=GE9kT0Rv
I|c0iW+9%o*7Rm<0@!C%2P_D4+59SinUa!l9R;#4o;>zP!|xp?_2oEp9ks7#o{vTC&*N4#gnU;Gerh$_83djs4}ML(NF3e2`(%1wK8~^S1(gtZK0_}$`^n*8
#CkucxBHxdU)eFI$8Hk1mXu;5Fem_mwzq!%Yz;<GpR-f=Dk0C$r>l&0q`|!B?)grd79`fJ^6y*MMzr){*?WaKSp1Y#GV|&-E;KwO&rJ-4@1C{bwr8Q>Ya%Wj
P>Toagj^=kia%cYU8d2f3&g>J14O62BhWIjAiHWFgnDP~_&>byxFQ<ClXEx<GUh+ZT_XsdMZse&`}!_sDb-GCiyGr04_ocviDYz~;Hp}%Ou#LgP{)d62{__Y
7{1*e3|+lr;rD$5@H_VrL)V`Q%*AM?Iw=HUw>jqnePS4@TW@n0-pxlm!$l{GWI}G-{@g&5>W6;LKAC3YU>p)UD!J!=JkClP#E-s8g+c2$M~el4qx?E&i9+)+
lqX8n>+XWU_q%1IMsdjF?^jUy6Aey=ABTd2(~w!C{P=2|3ba1^H_-kl!t->!>dSdi2p2Z-XE!OpY`9K;Q)?XVb909_Ci+2Tzr}?9aevSryd*eqAQfySyrq(Y
g&+^HqCH;G3W10dDkLd+;F&T|?}je~N68aXT`~x-^Lkk|C&4Gs;-V%=G^{uOxyErO;89k(%k+~#95~~h80le+;{(+9O?MJ-?W?q*N<ay;$$xksvQ0vDAQ)bS
l_TPnnwe`*4y02aU;4@x3$L^*djsMKK1OdfowpnahYDTFwskMi98;z<O)|%Dze9<$W*ClEQTjG-DkC`G&~l(M7?<T+H<}Z}5p`v&g-A6K-%IvrafSsWyiM;q
r+NeoGo-3N?X!pHc;3LEPdRibTn!@JqCxY1K6|UH1RoOgd|Jm7QS$EB?C{bZB&?BeO&LGI2balepUEItL_g4dd@&nd@{~(-12H&|wv&@gY=Qlxf}+uMp&+lh
_>{Lb78bs_l7k1G33*NIAU@=e&^4ocDpI%qBkDZ7xqkoupOK^?E1@AHGiA@*?Y;Njd+)usP&S!~LduG!ls!@uDN@<AC`p9+y*}ss&infpJg?VvJ?`Un&h>o0
aJS<I*<I%tw9j?tNJbF)S2xY;vn_!TG7KJZthdAPu^OB2u30dQ6P38hlY_AEx}?~><v5i{-;HMx`1E37;Df3uivKVjFXkcai{XP-1H#;WRyYx}C@z7!;hhnz
O2xQVtN8PuPdfT_R^|L?2<r$+lO}4;LJCcg#xD;y?9y~F1}3><*^U|Yf7219t6WivFq}IkQ+vZS5NEpwdoH_FB6os|rCq@bXBxRvjpJfrJAB2rH7(|U-=lVy
ew&YSYaVx=2rKlSDlytObP;ip2giD9isApoE&B6=Y-A~(7TM_~?9;l`_vNZEWM`b#`Ethxb8Ivb(}jKr*l4)(dm;yFyd0iSHSGWM?+*5Z`XNTi2zQwoW%?R`
(W;oZV8s%M{amt(V-3LqEpO9?cMwe2jat-(GQcjvHSDbz2ASu@efs;{ankTYO39WBKCqmudP8~{$F--~YMOlzT5awTe>MhEp5h781b@7sqWx>ECIyv()hkgt
K`^^mbAoQw9VWCKzewK(!btqB%B#07@YrD;Ix6Z8->SbeLhq9B?>VV-?9~_y>ruKW9d`!DvPw7CFGu)CU8S>=a=^A6iPq@L7?2;@FTKT{gw7^0c3le}d|NFG
lrGQ%<t&9`*c&5w)KL8tqL)MLgEPSyqA9qWd5q+lwg+BE9@BQ7i$$R1y!1KgP+Unp^!1;NJ$5S9I`qEYM?hzZ)L@!3X1+5VF1_LpxikfZV?>6KU79$e-W`R7
)jd*^pR#bus+rV%rWR`docn*N*u%WR|EN{4ByM&KRLWSU;pidXPk$NH5LBXaP2fx>j2@{e`0#jxis3Lh|FuL!Naiqq{^ALF{e~#=1RwNWVU};VkA+8c?gB}6
Dr#AF&a_LPNAv3se}w1#aaia0#G^C|v~$i~)cRlm3aurBe@FSzq51gnn^k)p*6?*2dmM(`!IP)yUs{6go=Upvs5idqZ%%S#2S6n3cEJ(42ngpYC>bvXV(;`%
2T`|lQ0DG)_|`}8179wm71<2Dv=Gxf%#erxZ??@wc3*;TN{bCW2!$Q9QC3roJ%}^qKWtljgH^Ax*t;wV8G{!)-uf6|*q{2dm`x0{u2B<_C*;CB`|SJ^dVdIO
zHc3$3k4sCYW|>rG7OzpaykzeKtzuGRL)=+CRL~&?y-u)yzs((^82FL%fQ4?FA$9cxl0ukmFjR+FSFttwa2xF(L>cKo*;^ex$Wx`0~JT=uWvv2;44wat%q+t
F~)MRfoA6-jJ_2qXKjbVvFpGa3H?}X1uio%e@KG&y$Sxvgo|*wmBV*C*Bit&4<|ZLgyQ|6@WvSehn`}3_1k<p6LF@xC)Mw|;-Il)X56+HPE2fEmosw5wJfdU
$5>3!uwV1h>-TY(Zq1R>|E7;t{qWFKLaw~OGHS(~pA8Pi$^Z(f%UE>hv~TdsL33RVjhSOMsL5|j35EC~<WSK{mC$9>U7R!$3k^UEUuTe{O(LFBKZqV8@&8Y!
9Gv9OZxGJ&%h|T0v+-ewc6llHtAfCFoG-|Ith4Z0;$dw&r4?$6&O2J0#-e}eof>1YAGn9VwFaIb^tR2DKO~Hb@M>mfATBKo@(X3eP8xnFV^!R5&g%r*!%wz$
_M1X(yFzxMG8O+8X}|Z|1Y-XyuJY*cDhyZ8UHn1FhdbTjBPF>pD9NyLeHAT(qwM+go&ABh@}1=Ar<rg_EOkDlj!%c9*UQWAq6p{0U0Cw)lP&ahMH)BL?C{1m
Vbnj|2f5pgvI^E-c=@5KK=f`R?9GUudOQz-*WYo;^J{*HVoxRt+Ut$vqcZQ)K8EA_be4!xmOj4xxw~R0Zx6hGCiwNMCGs3>9;76>BJSSokrVWxc)@hL&biqQ
mlJ*WbaDCMX!K@cIaMAkc7$(~4LM+mSWuJdasc+R#?FN>y5JoH`+XHjOH6uuR5kILp_xhLUinp9c*J)qaJu^AzK_pmAzE96%LD~UN~J?Serm65zZ<p(Bvf@s
lRz)5$DeU07|R0rKEYfOIIs6*gQDLR{kM5OrbZ{icckWCXV4WWZZqDU+Utve+gp8689``bVj7#|;KMwt&r0^LGj2S3dffSfFG#GJ{d=AGQS^#CvDDT9Z<^y&
d#X&4^fEMMEh7>#Yf-^tAs(>oSMUDg;{(=LZ+jF-%rX49UhVXB7}m^%LKp%qAhq*t-*0ahY_&%%4DtqGjfRf6*vba?x;>5=bp^ulHS0CU7dCkFbzkO^odQ(S
gzK4$Y*8Ng=WQxMCk_ukx2qcwxWb%)MK;S7V&7|(FaLJINum?#fv0kibosh|RDUQA_Y3SXpe;k=?!0~sO90e$iOLdoV)1g+M8@@=JwCoESS;!B1A~&_GYjhs
{O})1CEE0cwobsgj$efSElZ=saxV_c)mfty;l?OSvi;@l69e|J@K5n1u23VpsCTh66r5Kmng#}qz(rzQquLOJEb)&lrB@0358uoXD|>v+t>BojjsnYkhf#5~
J)*U5wGJvcVSDK26xY63u=%#tDQw1qu6x3iHQWGii`t}Wb)CRd*{Ibc<V5iMmDGQOE*RGSDj5~xkGey7$r)Cfc=^S=a%bEdNmcz#eP?uG$-n%e-bftB8G{_T
_6LGJa_`6QgE6q$e39eQXM^Ey;uSo9!Vt<(p`&-!7p}pJ?tj@bvFx+TQq^vVh8M?7ILHY6PO9M1m}w6&4t|$nQbiowE|z2w^M?J<D~<RaEqqHJXS>#{j4)Hy
&a>ocP#1G4;5b3Z)8$?z;qVmPs`T+wST)0AhEH#jZ4}^do+iIE5`ry{cFUYzTd2<-?($HG#mwobmwf^HIC12jb>BIEP$~Qx(#o}gj%ktwEy*QBo2B;6wwPl2
pWo$sIWgE{m>0KW&5wufYz5DQBOs+|#}u*QjX)Doe#hU!(D_%b;I!omvTL`ab?@7Nqlwl@fld#ze4OWeit<4Bnvc^$M4!+HNzO8N68NiwgWP{R5X$dIw7;z-
f^m5M-M8WZ^oAYqoIYfav4)t}i3eRF_?x{Ugu@L?l-W5yZ>Hk#b@g>e7cS67iRg*Ek3f;8%}gduB06I#RP24j(9}uXZ9y4^2($Fr*Jq4j@HTyen%M)l!meIu
+~*1U9@=m1TXyI(rqvbxaS01u!b*jkM!03sOk3$@jNH@3%=KrJ@iOhPv(jn8d5d1-eEO5XNg{`9NW1KD<re35u!je3mNLbLF(l&?`})}Mtq2Gf>I`->ID=~C
&vRaNE8M0o{(kR_FG4Shc7F6v#L06%%^kI_A;_+xD<_NhKmWeK8MYgLB?Uo&>m_G~oNzFnPI`WvFc(gLzg@i$h$=JTAEv_gkSS)2F*=<B#T)Af9xd9#iqE8`
Xh0XU>(UECp6M{s@zpy+&~Hk3e17cya6El(*YYsX6DGDd8Mm{OaBexAStPj{U4zXn-zHm8c*Vb{GtUK&S(12r_ZQ$?dhyKyA3p>wtnd~!8{+*F!;B9fufls*
jMRTR6ji4ZUOwmWfvHMK3@Mp6>_%?PkhQp?!qn*A!-^b8KGS<0KXMg?cFwteGV$<pqTh_kazU0z8*A;Z2FP!#aNF*;gKhZRW9RZN!Zzti6QgDvD5!jBNO~{A
O+$QibkPkLuZyr>eHjma>3#0!J_h4xTy1CpM>2RQtE4SodxCYKs^x^DEev+9ZM~97$Gh0hB!Tb*l)M@0<MxXI_tcq^@lOsYH%~o79AJYTVkM1(h9OY<L+(`d
C<Nn`e?2~vhQWsAjeXjQC<quRwibTyKzzV8oomBB_&GRnLntv4XZ+$_5^Zuon&Ur{bubkDsg_43e#)ZGX5!{2Hb1<jzt1>69{_J%{Zb}Q2Ut5?`K%OSjHlUC
0<`IY&=Fb?yPaB%93R)no9}GE^ihcAc4R*ODRa3DzOO~rf<C!YbQE66P1>IE3Wu>%=$gAt5*noKG$9lMhBi8D*~%<b_pz8AzU_zRi?472nGinnBxT@)15R$y
o%Jy>#gkWiN2>n$L4RI8W=*Ub+vO$ow9~VMd|Wa2?y*J*?_vF7=8K@Pb=%u+U5Ou~#!r$n3Q>C1`u>>$4Lo7~C>%%Nt(_yMXpTLJM)<huAy$_%7>jWYJ~g=v
a;Fh_8OB8Xy>grOl6x!`-aJT&vGKvx8oHwo9^F8iX5PRHnowBh8#+$(WneBqG0o|3BD6k-)RYkWLb*JiXZN5FQr&VNG!D+7Zmv>Q#VrODvH><t!-4R7);tu}
M$n--+k|dqChXPhy6h}d@YG1{;<9cSbg#vgr%T#kO;+`fW-*};?tGqNk+FeK6CY{dX@X8oQWg(*Wuir!H(#+}3R$f=3q+P?uxrm3Ap7HmSEJ=;yqB{;ls+f%
D6tVJQDO5rRs^O~=X^=Oq=3AdSiP-^kSjNR*Fyy^!t&_bBV@1gz#7L)a)K}7Kb^|l-y_OW;|n_1<e5go^JZ0l3lt3|<7-GYD_=tauHXM>Gi4Qk;$QLd8#~t!
GAu=Un8gA4<Ln|u%)anze@kII84c#{J15*r3h;!&>~bKRE4Z!heqx?+1eKD+WdC3mo=<$*$dw9)NM!7{10{|4&>x*)8tRCUo$}5Wp#aqGE!M2Ab_Pj{Q{b<1
1FVdA8OM_Ppk8IRl|C*6m3E51H)Q-_&*m!2#peUw>s3ZZgkB=tC#BHTr33kfcbBeh=wR-*#VND5CQyoVW`7su3(;STI|ZhG@Ra$<%J1X@W{WP}B?}{nw90Hv
U2#M~wf@ZnPFw7^d$aV?(-&>UCr=(|wE^RSOVeeiO>y+qpEuLn0igJ-A>_Ish2J-hd)^?@gv(TC^E9UoOs7)hUq02xjFvG^8y6ve+RsU3DE_bCtX_RP6N%(Z
n!hsFtszj7_*yB+83n)ZtOnYq;mN1#Lm8h95pucWlf`L2oEq5OLsD!38e9I5__zeb>q&&F5I99FQfFO!Ad-;VG%l=eTHtXlf6RF<0HFi3a=guUs8k{S{%74B
iJ8WC-zGcaLqZQ1<11mD7<7EM)NKa}K_c7Hs1%Tvxdk_T<pR-*k6L%Dolv-<=sx>R54$ybPH$o>k$#o3z2}AvLe|R?vj+XKeOqUj>Afk+Bl$i)7&Aq+zsSmE
8d+q>+gO|?^fX0vzE0xV81PyhnhMQwfXc_7Q$%_3;O>)&ZhJ&n2Sr_LDZ2{ZQQkRhq;7_8N@wcosYEbAYA%{B5VuZo#?=&>!Mb`|TjrN1EKPPe@3KU|<+g0-
^d1wW9HvtIQmX@-uimcn$NgaBNIt-yV1p-rmkQStg5aP3CrH^R42@o4<JOt_*f?4G_{3jpkbH|>8=TU|g^PyPFO&UIPBhfFNG%R7zdM(nby-4vSh3H^JrMq_
k)|x<&T!As&-0}aLf4pxW5<Ue?8%E0R`v2k<}B0KjaWwzox09X(GiOp+m93jJ%m0KnLz*Wk1lG1v?Cf1xI<&RN0>)C0`%=M-^+{)@VifRG7xU4^7xc#(Pxjk
O`jH}bw8+{($$TKPD6s~hv&>n>d+*sc=5+19!Ja-ITo7CK_t9#LwZ{ScOICCyd&g9*0b;N^x8q-`Q+WQx}k?5^Q^$iG)LqbX0eY^XF!~jZ@~4iFTR_&I#_i$
que^|&eBLC$ZwtNa~pLB>lov{w0avvAKaz0(<kuvgGU}mmaK4w)}Wk!hLD?g8&l#}vOym&9m_2jjDGVoJ)Ysd$h+=!NO{R0ze*aPYqcmM^Ar7n&%#c)RP}Ov
%+(oY9Cg&KMb5Cf5qghuS`#JOkx}}Cfw-=EnkQ~M1q{ZfX1@F5L49fQwb+t51k|ooJodN63GL<Y((dP>^piF}RLBM=LMQL{J0@cJa73BjAUCct*01OYCqkU(
hXU`oAr=ki*Mm5wahWr8&s{$+Sg2&~ap*Sy<7Y{o*(?jRd_PoIOyv)9^W~x-0@qcYr?RiV6NV@slFO`(u22-^o)Tj=!TTkO@;5UUApe^DdPFS*3TMmR{1U#R
RH4S3eaaU{Maqu`6LRV7=P!;|1%pt-ZfV2NM)3DX?DSdOML0#$ncE%{2RTz82~}@Pgi<}8B^OmhN_`sL(4r7FG{fy>Wr9(2D~P94#SaY~m14R4m*Lj`%|Y<0
4hAg*&I@yR;=NO{J0rOTY{*%1YTtUmFQ6nWlfeZ=)9gK(S~ak4(`}<U6@=2w!&X<hO|i2r8oII14zm<(t0cje@NO#E{=J-poTJZT73+om^Y2q^UGIYD((o`=
R=n+;6ZU)GavA8gz|-HxylExo_(fduk5<YECQ&qw3b(zG^W}@~c5Vd3#7R#C*xTZ5Vx<|avkwgT!{=7DQeiDva6xZT6M~(+{T~G#AwGI|pFQyf1nC?q6cks*
yTo<h9$77X&x<}1dCnD_G8M;m-rK|EM8d~5g%~_jc(&=qq65nL->tV6ZPET|@B-Ttz<qXDgD)oye*K$#cQgrHDDnB@eXdGOok_TT=e#B!#%Wk6`e<Qr_x8u@
c7&d?HTW{wB>=`tFZ$XN<iNZgu$lBx7E-sA#j+Tk@kHWt;dn$4Y}4i^zXix3sB@2G?<FCe(<Ex$uJA?sfhuX*5)-tB9^$ZkDvQH>?Y(Dc1@Q8xwypL>CLHSZ
RsND~f-k*FGIsf~NbkKb_fJ#~xlDROH@fBERKMZQ)ew(Ip@Fk+{;I&?<g(D(j2QN<El+hcaX=v3{&k8c8;*+*t1TMJf~EJpuGWMQ<eID!&(mB6QOZj_ok$r>
n&mI^hN&Qs!LRTUsT*|c=a>cgHE~gFPgGQAHazc5-w6F024!2p?8T{M=-B?0<_(I6ZLfmbwR;MvUy%HF-%Jjc%SBOKzEPkJ%RcmH*A}5e+&_+9Gz7KXwn)<}
8Po^s@m+CN#0o!7pE!2`#(fQ5_HV60&(svSyC8~pAx1B^sy$$P&&N@HjRPi04V3-e>WJ4;7=J?Ufy0#HBogH3K#{Jj?KB$zYSzOn&Z{EeFR#wCE0@R6Y0;|P
He2iwcJBRqQ3bq$Y_;Qt5qPcN!bJT_3IoxZ^ez1SP+|^0M6p*I$#n9~c5LQg3l{UGn^Z>jk3BV%9I;^jXQ`8=V+gruiAQwzouN7jnltw#aqJ><k(1~pJkvdA
Fh?1JeKnsYW$UHE(6VH;Ka3lE{S?P6gioQ}ku!U8%m`NUujSWVba7x*K78moKb+JD_s&v>W5v~LK=`!*SpR;FeDxUMxwVpEYsQZtt*(;xo1u{TMMvAw<&019
Kl2XpDq(CQDWrMU1iBtE`#kIeKtf;p{DwChnm88t7nxLGHu+UXW=0cSl9K5IES|_1+ZK(t2RNC({HC+=MBmX%4@o!l5%Eco`ro29#v17;`|Q=A%;kN6KFH=j
ow_dO^*R@s!LRnTZuL!hoD(2=??0%3J9igwNlp^0e<k&&_;e9UGxo3ZsS)JWb#Extb3-RG@=?M{6qw(B{B&q34iV1jSFXCL;ZBccy0j)gK4r;Yr+&kY=e-X$
BtMDao0e3LvTr7yZ*ww}vl^h7IEUBe)OoyLxu5ZIKp(7~iJ4TK+>jY}wfaVEh!=fwQp<%&5I%H8?Fzjm9xhE9n9E4w!#Af}@8qSivPg~@oj@3lyf5ZHsRk__
7T)Y?C3H~*eL3_>8qIAF#2wh<uxU`{LLH%q)vNV6V&77ccw)_s>YD&gDn^bM?{~&8d8X@I`?A5qyB2sd=zsTFxfKKo{O~kP&Cb6z7KiRXJhR>}gB7;Ph*eS<
c-|p7qx_Zw(zcAZD$ZNtPOR|rEM-f4tGqZ_(C-h;AT4$wVn0N^S3JKp6oOZUvej3~Y~U#!oN42lMVP;$zX$tFz%IsS{7gU;Qy<7p3eU@eGPT_Ls%Z=~tCB{8
$XsyNlEmQvZ9IfL5>xFYlF*sygav7N*sSk}b}IWoyJ)zmSjYj#lEoxbxmCgaEb**Clpq$~{$}vA)xx#@SI;jm>EL~Af}y;XKb{>t@In5L3_`Q&c!+K7@wuqi
<(|$l?3ErU@A#$z5BA&LLo%W`_~b4>8LJBZ-HAJv*+LJVoM5tLg3sjHvP+(OC4lS9cQx0%-H}dKwwwIc2R)G&JgEqIcJjE*nX{w$$h(;Oa5}^vA)h64qxn^!
l|QveC#ZuD9DzT0Rs3=3@>Azc8DB&+U2p#}?T^U&hpLord4clwy|cvm7a(=?C4)kh5K_K2jM}Li;F;L*pk`-VFj`(3SE5pY2CGkNCvN~=_uktdm+Fo$8|mve
CdBdHTk=-tRbQ+JxBVz!G(nIeo#hV?ANUiOz7<cW!4fTb+V^ilxSTN@d5uU7+4ET$-P;1NF6iX-YiGu8uc6&;@m1J<K3*O&ABj(C!n))k3P=$9K)2YZgKw!6
fednTSl6Tx|Hvwdovg9BxKe54&}Y<3O*%na=9#o#xB=)Q-5z=KCgNXy$Q`Ql%-Bj4Ouc5F16QJ>s|+)~u(Rf6E6f4Zg#B}-pIAZxq0|nP)_BV*vyeLz2I|l+
od-*$5c~1;{F@~)Jh>5bAUa6^J#)mexda`5zrL*EC8h>nsmI#XH<U0E_2t~xopLNX*3~VYJd0+>TJqJ$3K;x%Fp+0ZFv#i_xvYKMA^BF-WMq0j<TEQa>g+jj
p<?lPHMs;*Mv3aD=Q%-B_UE;0vpn?qtcrxylE5$g&>NF+Fy_ALaq5W=_HS7`w4E?R>XfXsY|D9!Pm>3oBT~S!Vc}P%C00CfZ1Lo5$;QPs>$dA*;s}4|b0gM(
;H!b8YlSE6P>>#biMyPZ;Q!8#<nPL0w)<b%5ko&nYqZWNQ60zB1^(%M62VYjt9_mqD2UCmZ26&3S$r$gxhQa-6)#AK_ev78V|i?@N9!{`bZQvv#&Y~o>LC0_
sMQ>6olU#P27sTZ9%<(q=R%9cSbxg*G?o*8XKeDQqvMaM$L}3oG!K-n{FtAWu47C04r&XB0_nHT$Ye!C4LvKQd%+E|d!=&2HEGZn49nAd!i7f^n!OC98hBGa
wHr3#i#pdo(;jW6;MkWVFe5Dl%_ILlWe2)qbC&h)>Vz!hXD$iF`Pd-sr)=ZOHaWa*aaz|G>Op&BR-i&&2><9GC%+_cSfL^nQJ9Y~4*1%5kp7ZFrOv(`uV6vc
YKF`VACg4~$)d?5IWIna<C(1+@`K(!OB;JWH8_0f5}<38|G#@5TWYqcn)ZktW$lt}GDc^qw1*YD0QO8?G@I-eMA=(sccu}2jBOekr{xOabfT$vHNm%wC&er~
ulB&k(eQ;EWhC}o3!I#OeF1DO{N&NZ7I+;lIHyReMBux1$1C5B;h&|o^0CqptIA!Tb;$`3GneJ-$cx3_DHFBXV}6+XzDC+v9*ZmeDb*@Z5<%O(dV7CXAVQNe
T!W4Y<GhH)SW$c{9`28IX#G<N4OiNDgOL<G<MoyhAT`79w}RAXe2l@CbSAY+E*i(WdDjD<+F|F4^`C79eH8rqnNj_d1?u$1bZk(?rg8nAeTQ#j@zBU;u1H;o
@LBX0IGw}XxxeNvdId-~Do+%u_rW~#@lA6geJFLw2kD&_MhjbeNH&`cbgbvC#Yb{bpU(ZTOf?DSInp<}K1g7g%*gTXSp{6M8c@#7^h2k=o*ZwAEN13Un?0|U
28GbU;@~NH{8g(}vir`4vw@Lj`w77-b+%YoYFY(4yY~lbAH`xKDp)n?r4`nWUQj&Fq6oLOpD}!8qo6IIoh~;?0n5~Vh4N4Fa3v;Qv-%wYISu}`;d*^E_Lh;Q
8b`pQfqLd&n<_e8SGl<vEs$XD`ZG<(2^8gFU1K8#5a)k)Y+6DOYN<gbQ7ZcIRMfQjARr6Ut2$+8o@-z)35#TPz8|c}NV^WoDM09^-0|*DrUYGV@ld4-LhYQ&
ADTfals11Pb9{ak@n?r349^LpNpO-Vp*<G-an%$IQs?2M=Kp)wRRcSDb!#sB190)ihf^ooY!Ev9u*x-627>*Y89(e~p<Kp3w?d^3xzjQ$#}C>d+&Z|~zjzSg
JaUduR6~U8g&)ajF$BNq%i_p2#OYJgX?yIXLEfymS>mAyV~w<&0zZPTLUR7q1~~w15!-h}MX(W{|4f9{3dpUq{2R-P?aIrQt13>=ecMQ_V6B4pVJgS@8#$n|
QLh!z=nU!nQ&Ph3&A@&|;-lC}7xX7;#%&q!LqL0vP{f2KC~5o|*B`jypN*M^YMMOc<|sENQgji&aKU){ls<y}6mkj_`M@(d{WkEC85DZ=R0SJqqE&WXH9sH}
<9Y!(vc@*39w-|;N$9OFzHXAZ9MmLO<u#+>GDV1`noRWn(nPJ}&FbIJEdSGKSnlIHPZPDFn)7#5X|FQ&4LmM2$}+;-$tM+TF0$CZlo()aOvuM6$A#r~LqwdJ
_b3UI#n(r|m#!W&LO9o<=2}88%Cu(NpDU#Y{|S+w`iC#zRT~TSvd1~Z4E}iOG%bVivm-~3%M@ZwKqXvQ#1tD$bQ`wyXVE(Os@(pGGvp&|Zr+j*0)M(_%yxo5
c039-sa|IwE%JhSBd;xzs7rO#i<I!{ef8HF3IlBIIr)s_aT*FcW%b`F+2JJDUtR%C15l95-u{<j0OJsGvMqTn=p0^sdMC&bbP<0?*+Z;xM>2#m&s`jDlDe0(
h8=PH4g==zH{h?G(9_PybbJb-&fK3~f?qPL31%GW@M}KbO@GM*!S~~j?GpN^gM^;4Xp0PLS9+bEG!Xhr^ssd34L{I62%WT}_dwxaYw12E0*@%_>Wv$l0_hop
L~}718SSwIT^StD=~}O~xMTb5N`be9JNi$pxi25IK-9)eIselb=m`bpNA%0V_LN{Kr&}<DqZtlMwgy34!IQzH+YM<Am#5TiW1$t#Ec8VSI7Tu!5E&YXb;cN#
X)YyfDSV|Rqwj|NiExL?PuJl0$DSomCjd2O(WZQp#`r1l!ILF~(Aiv#xG1T4z-Bg0boQ(nR%4IQKUmX+_Q@f(lQSBq|5ih}c3lp9PudQm!49_{<P-UiUclGO
Q4^dSxd<OEolQy#g1_>+hr&Hhm=QSm-oe@*e-E|Wzp%fA2wu_&`?C|!t|C7BTg@D2Ri~^EeT_j)&0JROW-M0ZSCT1R!cY*GJ^Izb5BDz~eXu>G3Pu_I3Ii)e
*alM$N{HyAndVq-|A{~(X*@ic`_%*y{Nl9Sf8ubfb1Ni?C={c-Et8)-<Wa@XLbU2{4nF7i6rTdz@!PMTzq!u|zAb6X8a}6?o5^wC)x{VsWAFN>&IaLN3Ryx8
6QMV^n@S&b6Ghw3=gsnmez+U16`D`r7AOC$?t@N-aJPFF9~fzbveFyMrj7<!xw%<M9IXrT$q&0058bf1dFP;4fei#N&i_`oK8tk?A;TLw=`hx$q2WpuN4xmZ
n4@%tAR~HmZTF8a7{x|j-!qbizz?agE0hi(KlYq1#F#L@vsXn)G{#^!Pt{5`piSWUUSS$L187URh8b?yz;9(z|2UlsF5WL1JRt1^<de^67Ad0Sq~D41;y46v
@1Li*cOKKBRt!u(tU+4m%J#;#6z9wRvQsK_&{mMBw)TY<AEvcj5A8Cd#fmiM+g*8__l@S1aM8k8ZD+-C0$+*rFZk*?rNJd%H9*P!4I~!pzMK}-!sK=8sz`B1
(3bD#7pqXjJ?}b60SOU!h|MO1ipoR0?sM$qI>4u{U;HIk7Z#?L)EAyz0pIjOtg1sedQ{7IPQ;(aj|)k?cTOlGuczkxLR%`*60)?OQk}-b%*iLUsfkEP+}i?<
Qy@w0yg?|V=qBdqE#(!4Bpc_cWA&UEwMjg^MJx&-x))6Rg+_QA&*r&9Y7gok@ax(mj-=Q^{)0^pATD@iJ$jjtmt-9hrAqpE_wJ?2duk;-o%!0S8n7;1Y)C3)
+<grXou^JI{WZg@&CQQx2I|;xkF>VRtU;ozNj8Iv46f5gk=kfjV(~g@u^O8_{HN=CC7t;(D(=%v0yX4Sy1aIT8L;b~7Ram%57loShsP{&q{7%Y@|qxYoqx-Z
uA9Ivm9*fwuo6Cxu}pL=+hFn1ksh93vPg`0G&Sm}jESkBU*rMjAbTSCd&-<UewDuabl}N(JZ0$kyPrfA=kNOkk9*7D(4xriVk5=>{F_TX`I`GRALu%?Iu!+~
LZ5pi{;)lvmn54I&wiDMXR|e{O{g!Ds=YGC@2Frd(OR5HzYvM_fqm8o?t^Ch;6KszcpOUlKHp5mhDFZhvj?BXWBA@y#QGh3NGgS=Cpsq~dgz8{+p9;AoJnmo
;0Q$E{@$zUls=%kF=h77whUB<KiuYa^aaKK7Hh+58)$zbOUfYRNLQL1WAc_QO7h+W{mL@{GmG%iyFscr`mN2EvGD>Vf=>Q8Jn9SOhYKx2Nr`xolqG2V-5Q>9
G%W`j3B0E;9xSqM4&x;GS=5}vkC2m&m6m?EBUtos<4Z7J?KCmJ36I2_rKIz2cRef=zPeqbv;nsk&7<gzK$xGXbry<9gvvXYTaN;+piZS?i}|ku>?F!9gYL#6
<?-p;f^~K{@uJn>Us)8&AKb_~y`_bm)#FEOAO^dC5fsn6?a&t9qgSa@grM`<?{`^!pfBx`(CrhBM|p}g-~B6aLvy&{u5Bc4)CV;tsM{gnCQqvC?>IcmIj&Jf
=71NfF>|^ztgvs@V6Xm?gdeM)vrF?-F#aK%UjLCDs&9UIm5?Kj=-{Rr@gszJx_@osUQ8~=x()N>UIv1)YEzn8SOa`2GP}vL;c)ECQyiR6LEi4`f6_8)AogcY
HV?AK!t;}i-Gtu1(0;M*`THQST9Z&J-*$rEnlZ!rlK|wGt0~pXzr*maVX_LNWcWXf6l&nKhALTdq40qqWa=@T<(yH2H*s2#BWE!voIE+0h>DQ$l)-bI+!f;^
2K_S^13~-Y3TeUtHauCCXOB7Mhus$$R7ZI2prw(#lC<uDgW;jnrF6njaX+l$6CHzkB{vJBIC-p}H<S%N>5DvuW{rAdd&rHFT3h<LV1P_lC#%g0W#c6J8mTdO
8v4q#+{YeAKl`_sHWT)Ld^L9ekU1vCb}adl8xZYUqUKH2jU89UEd6~hh}3%0UO0CZ2F5zwKeewwmeNOGexwomlr!DUJ`<jI=r-{tlRJ_+C8c!dobiIXp-Lj%
1eUQLd8tf(u>80@SbJ3ik6o2Y0;@W~bo49JG0lg_;VXOH8<K?r&u?=2jDB!;x?OX!#TA3MleX^C8=#<ytN*H;KUUN$c&Rq?@Nh}krp2rR%IdZ8gNjc7r_*<l
XUv~v!EwT`_Ha@uG(MDiJGy%!IZ@MP?o=!SnOtwFsn}pXNhxAYJQF`Xjs=|dYKGV2-{T_+kD$7x^^}^l3C_Pnv#GsYA#h5GvgL3WX3Mjd{`WsO%*^c0Mg$$b
E_qt3Ozw$;PXDZm*<28Ka`WcqiYp!snXR7K&OjAKSk>*HiP$T=(jxCBf)@!Z#X6q}90oO|U_zgr?yP%#-_RdC{Sx5;lqx8{`J+r@S`*i=kLxEGD<G%ulZHcO
AoNy0QZJS{;m_O^3132gJCmsDslMX|QST9IIfWo(brkOQt0<u7Z^1s9j|4p%eY1UXR0&RxQcW+sG=g_2&Fu<xeMt54^7Xe#L5?|tGTp=p76+IZ8SSO8mhIk3
`|U1H2I|yb;xa^-J;S?b2?H$sl;m~3Y6y!L(PTFI$`G&dI&UN<fImk1$6n>i;}_XkqeyK<kW~_;>TVdrG}A0XEw37>K?ZILemanQ?(20fLljLt?knk~_BfK6
^LNO`2-U+ISyjPK=-_)$mEHRc`^+f2EpO<-A-#YmBE<}IeT{ziI0;TqIU+7e=>)Fm3%w6NpMbzc#T(-OhVb2pQ#iY*2clzYT=pB3==e5waKK3sa_fZ~CzFKn
^4Lk4(evpz{?_S^!_NRH3d~=<Hfn|(;zil#ivcj(<;bykoPa7`BQ1~9x)8|EWlL&uK^*tqx}{1RLLUioeaEYX2ft_;A2;aX<Xd^|dIu{gBt9u(Gdc%`{tJGo
e6monpFdh&W{Y-`*z$}THSA|O|1t1~E`AAi%{?IG5^cwxc3T%)ICPzT6ApVkizMFZyJic!-!|V}XzcNxH+N6*!*txbaqa4Oz9l*X$5@psouJTlsajH94nw|!
QH>nNu=P$fp0$ccd-+UYbx#ne3Lbb{kGkQ>vX`CXA1y>JTglQBdTeUPky{JG_E5J@I$P3X4DXt$L6H*yU@{$^CgReBx#sxtGwV1I{mp+69i|8El9pqAn|DFK
?tX+*=LW%-T>FMT7-4Ycw|f_nG2|~d4j&*F#R#8=fQqC&#)nw)rC&#Yr9PW3Dp(Webi>wPoVJ3L&poED_iRYs%w>o*(8sG@N&%V*5p+FOH$6Rd44=I;FAOr8
!ZtUmZ9C2v+kKTf4Dvp(Vq52pP}9L}#`^g#!hF7BSI#TxAA%d#`n`@Jz<0rC<vMi_*oM<2Jg#{nlbJs{%1#1zhS$PGN$qe`<waTI1vzLF7Z^3N%3)rZ%2wop
J@QU@Dyvr{!GlGhr|OIkR(W(&#$PLAg^`u9<FEvtwVk0pUapEVFDml1Zh63nXrd_26t~Rt>aWZAVh`iO-1now_~1kDaJ5Ydm9;v-ALH)AY|iI4H<<!{TEyJ1
{+ooR0rD<!_rsvO{KK@VLJy65rX3k?>f+4RX+!o7L44dzU)O83!<B`JJ5@KLz(!0fOI_`Vds8-|rY+CVm~>`?Vp|Wno7{TRUwqJgcVv55Ruh})sS1kiHn>B~
b%1Zn7ot^jQewNBFo-y+W9#x19MJ_t$#T+IkeLW^dSwau=>f_>o|6zf84@r$XNUo=HH#2&UEIFhRxI{A7~h&Mq+Xx1!L4~>vszvSJm59VsbLPoq2fO*kBzlp
`10#8kFp4U$2|EVN^6NoF|k{g@07s&sf@4G)B(n6G%7;2UI<xYuBsrj!?WwB+1)LjutIIU*Ip?LX-l`g?;jFHQE%cc(>5n)t5ub((_BQqZI@#ukqunDe?DfB
a>N(*+gEC~QgAoY{KH44036q7px=7wfSc|EhWy#2|M@rB!TIu3izw`Vy6}3>J~Ko-saFdn@aVnTSu(E0K%8*CC0#0~hA$8HRejmj#YLVo^wS|)pm9;wUe1z%
pwqoZnk53Cr+rNHc$tB<Jruv(-*rRkN~%|vxhdGkKDJtqTA|75^)Jqi1blfto_O=>GYGZN?pE{UU`FYadwN;}I+ybvpLdOb{5#7mw_YDeZU+p1n2Eqnso_kp
e=x2LX%`<Y2*epfhoGOb)`-LKwPkHP9KZA;o0c*TF&5D`C7P<hdZ}MLU)v2scRYVQE{H;5gVc-!xgr=tm><o@sGy{sn*0?bL6_MHr@rh2A*ao7=|^ciCRH2b
s!B^BdZF&~xrY(>ykIdgNn8Y8*B72eIfZbWyH`l9PRN0aiZSiej}h9Ydy3g>9F{Z2t=m_MU?9k{7Ln}+qNwv+$6B1xQM;ETQ7juu@w@rAg;Eev@xsvJWEOHL
MwjV&{7|am-j%!Th~xX4+>F+$aVq=F^vrKUf7!{t`T21XejE5T)Atl%&VJwOCTkWh6sl8nU5><=R&FI@jcRzO3`r}?r{EVk(fUI|zaTMou2gBVfyiZ<(5ulM
cxFa)flH$i$Db@(1^g_9Lu^m1e^exBs8?J1&ql#*D^T)NaW1@9+4WCb`J#RF`Vl+!ctmyF%GP3w$5W}WhPv!{Fulq9oG;ys#vHGdECNTybI}NE9E-ui$X~m@
1xHjE6n^`hlmYgMb76g?u{e>N5!|I4fE>H7GvmpDIKRM?$8bFX-FCVqQi+lHb?|<~^`8mwj~w`6qAU${F_EKlfrRrQGc<g@-GU2rDfLYAVR)GH*LJs251p-#
?Gi0gP;`Clz{Bt)gh&NWIF&Tuo^R&CMY&x37F@3Lc5B2BIY(au^&}X1f0er2d<UT(FP4HueeuOoh|8752lwlyub+%_gnWa|%Ou-dIL;HQrp*_R(UO9RBU2$z
n)=MhrtJgEkl6!NCV{9e)XQ(?%mA<NzGZ>UcwBBHG0wKSg87H5&mXQd<He()K91Bvl-=-3T&%eTouB9ZXFcN~bayo^%{&a$H<)-VI#aN1RQuXjEFS0A>VDUk
+{6>>Cc2+h0q~SkP50Kf$L4f?WHHSpJmJ~X^{dVkAz=aiPdAf6qbfW0aF6Hz*XehRE0=0rlCr^vs{X3N{v<qLrnYv^h{EI8cU(<1(P$N4v{9wX0utpJ#r*1E
wlVueU)mXTq;VYNlTp~)5iON;C<OyFvZ}|v#XvN`X@T;tFK9>utr%3YU_b2mb6`&~3hUEKSA7Zl8s<R66d!<Hx5hg>mgYEk{l!?cWi(dz-K;#N>xHAgqOQ86
IU}OoOO^YWE7T_U)r`$JLa-@U{|k>hHh<{a{Qaf}<%DZ2++S_LA22WcASoCcZ}yXtytf2PZ;OWSC4D@Y^So&?W`(iVC2r0RKd`-6?cxq_0<(Pkssy<ngzFqM
?C!ZEbe(inX2}CxD^ZIby%q?3`*SvZ)EtLKg7o!^43Yi5UanKh4vk*rH%+Z<F<D0?a`CJzQVt9cu=d!(Q?+HX*3$=t0+|$>aaJ(?kju?dW&`(kpGsr@%H#E)
zx0wZ)*w0j`N*?&Pn>=;$MDjM@cZskDe;u)qJ3O?EPO^E4~Hx>+{5hPp4y!mEpC7Z2j4KgP4mG-pADZEgEM@G7hmoSX&|Q`$8x4^Q7TG1!hFaEiZT)g-PgU)
POn!QTV)0{vq!3n>VO2_&G#RFxq{=mq%o7E4Z=^~Q;vKofYi4U_1EnUK)e4Hv7?AK-e_?5CrcEdb7I*pU_}=Cf1mERBhtgU?utxxX$=@xD!%XM49D{|{^pKT
vY^n-d0NaD3bFUU11xlXp~tnw`QBFxJyqhor#j4WE4(6hWhM&qmqmot|EWXrvNB&$s}U&IXk(bkoN>u*`TP55Ta49+Zhhmj#l+Ty=F}iZ?1*0emmTDdd<~|e
Jxzu<Oze52x?LA_4nwRPOpbWVcMbM@cGx?&!9rXc4ysN&8{b#P$ePWuc&TTMq@Xq7I1>kOU-ywe*c}B)G5$NoDelM!AW{5rlLL>k`o8(ODB|+npuID4R-jnE
UwpF7A7eZkf#!Q9aH##shiG|!WZqhe5Pf5S(+Me5-w8f+gURpG_e~2(w=CNpQc(fp(Yu?c5>4PeOFPDE;EdCO%$ZK}w&?u0)JB>ui_7v;S<ieUaa%3jO%Q6J
Cl0rj*{_dama;3WNj_*IQN2uA&5x%Ydq=%RTp%_gee~LQMMRoTM&!~<;O6P_LR|wlq?!;F^B9Dp{btnhYmanc)EGuO(`$wrnhSJw&GxuNz3vsno`$vEKEaCQ
U<5pu*SYT#hVY;KUz%>D;J4AfNu$$M_)F(SDG={~rtsgILCK~Vf5+{gI%ER^8V|TF)SyeHJnqd`ThOr1GpP>;p#F}ez`r05%+P1hx$>LAUB-C9<*NzK*R%Ag
tm#5=X0EpCrwvxJ2L7rGNnv_lfV$A>0E{puF>$}Pf#3m&ZL|D1oNIFsyzk)yuHG|rmhR@D(KznTZYl^Puj8rzXl(Ga#GcZ}#~Nu83}=ikNrUQPezs9tHYC3&
vFXz!K;<!yYUNT1=9+T``{?2kO3!%osdg-CZSBM8x-OwBO>25RzyhXco5aFz_+mX}<#D=CJd}QiTF8lcp@+n}rz*@F=WPd8=6P(vdq`X^q1G2J8XaQSy~H6e
686`|)C=RJI)}RVn`3g|(sNZqXE;{Bif*2?$9AKs%>_FXI9#{AvTHwujP#KCo&_i9lwLbuUn~L9Fh<6PPZzKvpfkTRrwk?L6MOC@r{mJvqmEaxvB<Q4&692_
i&f7WH-R@-@yR=HrJVN>{^W?Lj4KlO@^0^t)_^UJJSsE{a%%$L*GTybk>1ElWz-sAaYMtMaBT(ZNSu2(9r(}M1wY8Bvd*N&;(15OiXR2b|9$_^@rdi+r6l;R
5q-ZS<BDS6`ah413=v#1RTRQwi2KLS*6}U}pyp<Dx8H;fJV?`CeH9GBOpwJ?#BncBQnD}QHM`^4_a2d7BE>j3XTxvX7YZps=`nfVSeReuVkJ>b0QFts9#%qc
lKHvkX4~a>=-zR;6#XC%OgyX45<&wIOJz1vG#3I%dX6&sV}an*99HOFO~v!&Y<jOjA5106)ZB>)fZao}AWOn}e;&TnwHD}(>%AMUq6foZETS>zI+FlFpUn%E
pW^W@sU?7fGYE$l115&GtiUuVFY^4VJvbQVLcRUskg&1y9_NBFn?Exks~m>x<{O&E!wwLyzr^;&*AEd*_bwhyC2;rbHzRhd5L`;VJMU5GhTAzN!hH{PAU)$y
^1t^giJg-E9G%Gpsq7(NN255rzIICL&!9j26HXCNKG%kD=}vUfRt(k(j<Dc+5UgcBtPvA>UAC}T`N9Q742d*8@}2TSd%()=enVSG?3pv$x19)^%<|}twnQAL
czIHTHxX6VQMNB`t72>aQHIX(i=eF%Ix6}(3=O7{0e9-WFtyYsWhiNil4PCtIx~sb{B&IIpm`8fsvfND5_p@JYRYb{ECGz{_g;<Nj6z>#Y5&32n|NPtzEDJW
3s<9yT4U5Raq}%_m|aX9!f*L@3MFLXlGT+`mDC12F0qd45s$`JcDKQkjzD}dYk6-Yrw{Y2z(dMCF(5N~8hbP;3uI?l58q%41n=3{p(J$+P<JO(DTF7&pXRcs
j&~hW2LF9<dY*z2b}O5}r@EMZ^E#8HwGR1uVjI-zQ3wf+jxu_l4OOwh=G+k%$R085bP)@JnrB6aC8Hlk3~Ic-hlIm9djALG$T*NN#U$<<aD_@S`%iuXmpb%N
NX^(rqKknf>A|sR#2$;Dl_8J9=&~5m;K>+_{9t-h=o0|*ec2&2j~v1Hxk)JB!WKPq!Gb%piRh+$cD{AV7e)CY;ScwQB28s*MI*)=SEcVJB!2RPz=(AE$pCxI
DlVT^-s^ygX;o6mvMeyI&C%?!4u;?3jp3CnJ^c9PM-spgfH!mV4@uTDp<BnjDq4|_^F0lRN{(Md+8tGa+vjp{TlL$g8JlQa*&PV)u?RxSeM!>Kw_*^!V8+|K
^?&!^^EqCK%@+oM%GIx8&_5OXl;d{{sRMAz*U46|Bo^&TEKL=(erR}-%dYH`3tFieHq{e`_{p!Ab*R$=g6|fLc7;>WK6v8egRTrXq=jUe*!x3p_sh91YkS;(
^jrFftuq#lwxVo*#DdkW*3I>IKFYs`)O#NHgxBDT*Tl3hmX8gE4lz1`Do`q+CEf@RB<DvBMQc#DzTepK-96~o9r%@YG7}XC9~9izse$|WtG)oKC=7U-S?Lt`
VA(v9FMG)fUQx<ig)xzsC+7L-j4NQi+xBZ{J{j3(R1OPYNkIEKl{^YIKRB^J`P)4o5A8cYoNHhDK*pCW+h8;toz*=46a?M8sQpIovYdo3s*_h#$1cL-E%Cu?
x4fWVaE5W4#0EYp4t-w1VIa}C%qS=C3HdLXFTK>#QD>Zcc(5}RTXmcos~yQ$qoTNx6iv|k#smM@H<@@M;Ol1GpN*f6ao_cGLa-U}^WR`Y1&%Ce53dVH;3PAv
%Z091d=KMUyp)xT4-4k*WGPo+MJ3OmMC^}0q<;c3ovwgvcEVhLt`H~GxyOnM1F>~<bL4yJHR#$%cPK2FB6jHvMN4%ULK^fMR?@P-lJ@(}e2FdsN63%q$0i`*
%1rV0ad-Tv4roqSi^FvLQ(;rGP>`G+{5`H8jC{8q_a8^>@qNEL-LutjJR_nRJF(^qW}?%gG97s!ax{rOOqmIZRDsncH-B6hI+wRvlMmaTT+f<)ap3V<I>k%u
fDyfSFBPhNp?~-lMfB$kG`!9gnC?nJ#&T9nt(QM~gmynnpGpVquYwGk!!?l5J(?KCeG#6d1~qlLfjAQTotDuh8)21e`~N%*L*`|^()^G9IA7f#D-m`JuFF}|
!+MEmEtYSfj|_qES^MA~-4Jx^&Dgf~G@_QY>%(O?Bdm<A^x9_o;n$(6S4={c5FmLHyBwT?0~34B&QuxVLj?`PwTMInRrv5Z|0eL*{rmm-+yUS!B};ERT?(;-
qV~f@P6S^Rw}pTQ_*vCw)62u5JWzC3%O?Vp<;*Qmh=6QQT4F0p2p&zo$S&G;hl+;dCS`0hEH^#}w0mUX{-Nc9jsg#;%3jWCqEA3c&atl#jAKyzep+@j)fK9A
`wk@QCqqE{!0pBlo%j;`_3`wC12$^(JioA~g3+TSs6Da)+2NN>?3UcHUq!}hZYCUSZ%yl2M$9m+7x3;ey&v>^ma7Bh%s@Ar-B@}n5CJkVLevYo5PmHcP-ve3
lieIw<_FoxHuz*9xfF>kT5jfAr%TY)uxs3F5rA9NTsm8|e)!z6vOmJB5xN_jep3IBsPm4;>i_?Lq$GtRi4xf*5whZ8%ieqMz4zXG?^RZ2q{zxjQlXNfVI+k_
vKln#cYXf&zJGsS|D1EK^LjoX&-?vcxAVHTgHg(mQn*c%h|N8PHzvL~;mwCmtEI>~WLAz^hDb(2^nKkA$v`U{=1)H?{>B@D<<u95&qhIoWLG0nlpp{J`ktcG
Q83-UEtJEa3P0*qq7>IFFv;Ef^>uLwzU<AQDW|Z+X(re%Z)L$Q`rDrS77nOhyf<$+Z;Z0PoxlG|>BCxLz^da|EJ8J(O_DKWK%w+t+%$C@LiSNckT`kaXS;mW
-adWM+EWXxj=SKp<{1i=sVrES>b`q@SR2{z)XrYXOMwIZ{4=x1xoA#RBGVoBMXq$b)7z<ZP}<zx`N3C?UYB~-r!>A0v2F<U;*NlXl6%uxa}UVQ{3@b}^g%)q
U+?8NE;vv~@6znx30AAZ!LMgharSRocqd`rG3|6ye$xuZbKewqt`%Fn`t$9^kC8we+6vi|)N~tiXXN?2j)!7d$-v_4gFp=CKq~dFBf<NAgxn*{xsSHW=N`6Z
pwC`?pw2J@fugx$a$Q>ge_w}o{Z!xN5O{hE%5tP+pna8Wqs2J@EtEHI?5Rit_dLD*RbD@2>sanBZ0v{XYww(I1dn$87?Wt#>WI~s!dYz;RnV#0j*M}2!4R>p
-|L<n49-5!U6l64f|^`TL!A$p$=1_;=Hy_#=rK#9R~})#9K?>P2jWK_t3iBBIot<=5<7pzpszZA>m!dXZq#a3tEAjOke+w{Ta9pp=!NjM*Ez%SX(VZ;p(8eL
@vFSivxM8DVXkkqvG~=rcB$p{4ZKjf_teiW6x81hSA$BTAipnR{<*grNDM>Y`hIc5{tmYS3!gYV{~b?RFOm;h9e?@@(oJ}E<<|A5`B~`SA91`?HVfIW<=j=X
V&Q9Wz~5S`8JnUHPI8|LL!Hv<#KFU<@W0=0O_ZO2#K_mQkLFS#_oaNo>GxF-^`(5H&+tG%)n#M4JSEs`jjq?FWP@vw=*{!fNysoQxvZ9!h<*Wsz5&x2xL)|@
f0e+KzfF$Bw^J5?^Ymib`n@n@Iem8hKw6C>rAvm?&+^e5Bs6e-`U)%x_yVjuk})K%C$RX;8)e!-&pQkQkRBykE2pafm%*)>V+yI*d29GO(5emRPhN^I8+3uD
kNfLG4E|V43fWBgRtX&`6|Gj|6x=EiCC%gtfcVLWQ@VL!cz(EnGNU6M+@zcPMwg1PU467Vvy8wgv~||g+nKOA_RuA|BpwVA?8kqNgoB6dWS&uFFq+$=v=fG^
aFpcv?t_mmC^hZj|JON=-FA<3W<CPXY5)79-kOckaHD@zC`Bmk*M-Oza)`Q7K73Fk6C?un9Tc78!B<D^ru@7e?;_si%rP2)xKS|t;omaMb@h7)6xu^P#(h5}
b0PwWO<vkBX5p9F!Jn2f@v!7>Uw$H8hq41yVO^!oaAIir^dQn3W0iME<~ieWVTk;2l1?PbLp*Z_HVcr<R3ukX8-w$tdzq{})8P!FUxnkD*r5DW-)No<kJ^6q
fg=xaFPW|Fk8dQ-uQ+rmiN&Iu@!V-v!f~dJJI@jOdcgdeq0))DVr)kYxgL%61CLP5yOMp^!O(WTCEqU<Zi}%~9GA*r?EQp<vbY3H^iJk<#ATq2KUvwpm<~B&
&W-BT1jH;}>3C_D3rf3)q7x~_X!=2ul0vAj|LfEF44#5A?i6^FG=)e=m0?uakWNu24rymJ$Cp%VP<ivIjI3@h{tPYlDjMEIZMCn@Oi&QW<Chc08FCTtJ#aHj
pTJGitu9%sl_-^*e-y8g3c?WCvcFh|-HP<6%9v=FE7tGogeKzln5Jb6M>e>Xl&yp|;-Mdw^w-3u9G8zA5KUb!#v0f14RaYSIM!+Hru}tA>!+{UsWuVlb&Yq6
rH#j6&!TznZEFlMot3Q7a6s%cZ6o3y8@Q88IMVp|V&HX;(gRf=D2r6ADv`&-?M4pSa8@uT1l6jR`=TK<R$SV^V1;OsS7B1ov7m6yCP@gjhLXW<l<vAWWOmf`
@)qqeO||^B)XErNoQfk?-^7CS)8_I>pb~V_h1O($_@c24J#|;@;O|1u(jsGv2%$W-mE+O)JmZd+<JPzs@_K;gxE_8-8HUJ>SYhX(Qk=S;75qY0nHMxu5GS+W
jg7|-1E*L??>0x^=fgyzYb8d|Ho0e-!x4?To`@~|zn&oVlq24`nTe5|kE5dM0chrxU_1RJ0AJ&17cRJX;)izy(?XORj73xP{RTY|!+VsDwUQ4FBMNd_^$sX|
>nxDg69+lF_mr_WdC}2vN1iy>6|o+!)&u;y7{8hz_s+T*LWwNPcM0cfy{|k~a>xUArFZyb!))Mf+3TkH+85ai9LG*?=;6sd*8(mPU+gx|5Yvh#V^685YB8fP
riV$Bk3I-M?Dj^+-&DeNyYw6G-6M;qY7U3_pW5TpuA{C(8^Kp?!`RY(d7zSJIE8ZE9%2%Er@W#F-@_w9%wZ`24MF$gE)O{%_m;@vWD|3Y_NjH9(~kpxt;vQY
DWP8&D;pi-T){f|E+PMH7^1Hq@;ymr0q4$xWZ49dw)}HX{;q~2jwCHCnZL_|jN*d9vo>vT-P?Rcd?yVFOQ*L>zSu+G>((o|z(~|FB`+3-C?lPE_%QE{Nc=w9
-Q#HCg&8Zkv2vDVNWGR96Vy$>lHpL9{+&o1`EB!Zgw_R;9S70`i@lNkx92`Xv^QEGUmA$sYY(rP92%AuFI*}avnthc1yQB2u+4rK^yg$KCjE&)%~a024t_19
SLU94;_U~A$5i)BPnqG*o3AHMM9Sg|sbnDUS}dgEhbLYC=Aznd&kL4CYkWGVQ*P&Gi>T`JGlsdzxO@Go=Ok|d5+132k0#XBMd><I{<v%iIcG{A$%_F0gh8Ib
eFK<eUP`$qkc>W#j;QTvPrTstuleqr0se*WJs)GzU^pSF?)oSXiQVr>$5i5=deJ#PHQNT$;RQ$3&#7R$-*Wgctv9HY1>=d>Js|a=C#R{{6M2$(O5WaPsBie@
Zm0bM2R69luS9#|=Al<}rko9+VhJ^|@C`xF6}=B)3c*m0acmUiyn@{`oTV2m1K=EQOc(Yu3W<MHijBwRL9~77GP!gBSc~L^ZJyX7wMdGgQ8EKZ!asi!KG_1V
eeWw@rFq~@(YQvhUk3ygb~|4*d1LoZ{Qf=nqrf3AEOqR8E^0Qef+g7v;;m<QK8xgmHKc2w2%9N94TMuTZU=$Nea89}nKT})%Rej&^@sExzjn@=Nci6@oeSBU
jyJQcW0QFSus6KnVnv;aC|0vgmhV^5J>7R~$Jq_T|KbhXPTD{(OPA|wmkI>VvqW+ST7&6epUqA{02=dqvlw1Rp`2uX;}xR?p2yfIvZUprWbT(=HNi7esGHea
@7h8jujJf}ToUxc8Kh28v}17Q{zBuW08IReS2lj;gcDtP)Y3_Ba3mox?JAKET%xzU{={ZNwS-5unTqwl{I=^Di4_ZIg-(URu=Y_ubQkgW8_!umU^vp{sh<n>
ukC(eIB14Pv}Y%8Jq*DOKgoQja(6r!0Yk?B{>S3>8p#-e53kF_lzSLtfj7YXmTGq`c-vmwK057<7JBKI*H`Va@%5vj6ip5i6)PRian|DHffL6=3D<uz?`^-_
)jX)>r~jrkPQV|r822aj*7)8;;x@RNg`73+ZiOS!s4VYQRZ)w?RIYykQ<@t{>=?YL3LVh+w>*sfTnFyt&vf#6+{TfJ<%Y5<A5aD~FA{sFV}gx2WtzYzu_DKM
)V~BG*x4wndv82et&;hl-blobw&cJ#Y9X<he8MUr6rVYc9WslF!|{P-$4<>ete?#Ll-h6)<kDdll~p0Q|52AkhOY$s8jnfaGE|}U@YTe_WYsu-cV3SEU@{tK
j((>t2tYL|$2MSu3$ARRdS8`cQp(?*$}<<?15Ec@ZbU(J#pZCn+Xzx>U;Um+j>oL=NLh1M1#a#?bCirO2YrcC8keti<C6-_0LgqhY)ga#&UM*Chf!M3t1Tbv
_lgaYyi@U*ZJAN~suRpeQVrA=tPp><`GN&cDPrll2jhO$z~Q;j!Hevn7}BlEASQ~0_@C!W21^A<7Z^u?Sq_4k4)M8tD2LA1%`-3ZGjNWHBjJNh7QS{Lw%%`9
j@4dbF$1<(T)VKL>hK{GY@;s^p5n<wX5I45v92ueJik~fW1Io{h~#@hr;9Q0QiQ#`A`o-dn9rq)!Nb~*X6&h1a8ua(@{4{YD6_8qjec1HZ9dh6r423o3{AOy
*fA47sD0QOcZ+d7uc<Kc;#JfRlhw?hv_=hsXY@_08*nXNnQEg+!U^8pr_4LI5jkl@HECUrja6Cm&#BQk9O1w%B0Y@N4-);3TP?WHqVkt=nNY9)Z2L;x8zJG5
7V?WT2wyg}c*5m#pj%&(yC*Lhsv^ACI`*c3_TZTO(yt8gAbghlPCO=WFI=cx&Bo&RAqUcHdGP!E^bVSe@Z0*P;xhsde*Yyketn$a^X)EdW4#X{y>XgUn>q@7
WHVdr-2qS;F(P_=A{PtKByR+%=i}SOzC4R-budr~_~FUg4%d<@_xUqfxMn@$6I_;qhisbNC!z`U*d<g=N>YfQUmpzaIEMY#r$ljvq(46gVUJUsa2rEDTBYwC
>>;kgVE>*;*Ea<J9^YYR*Q&-pQJY{ArB=j`3`;WHaY43dqIB=ySkUhetI1tSf!^g)hPpk~c=Xy<F7;w0x>*H2y|TzdJ8OE@GnQ6F%$+vZ8@-NBBBSpv8<}9e
`{aooYYapcS^Wdavru^^>z{~NG~UVI-O2rEjbbLBgo(G2;E|Rvmzs(IB{kLNR6-Cw{#^V+bj1-ay+oPZ+XSC<%2Rz{Y=f*PItfp&+2f+85ZP3V6SkIo$Df7!
<EiU;?J2S#s6Gt2;P#a;$4id-=NMby=kP$4%0LtX@1Jb&kMIQh$dy!+N^fiu&uCF|IO1ZgoM1eQH3BDobY>@dV(mk_J{zwpPO9<5#QtwznNG3V3#kM@;~W!d
<aEZzq@;44&2ZSi?CWJ+bOn!T?@)i25fYBri8>g$fQ&?_Dn;B8m%MHyX~?JI3-jt-9SavcJ4#IZ_)`!f4YS9-dD_7x@cYW!CPL1i{FFvt=m*}T)otH1<3V31
uy%jQ3OX+y-O>9Ijvci(WfU5|FeeWZqkiH9UD^jd_X*r}?~Lnql8hwSx_0Gn7<$6{W7W*$^91}d*%IOUD}<JU60u7@9&pO&ko-AqfD={gNi{RqkzsE~xshT6
zWI{m*N<J`Kh4jk;co_}oUB()={~roX-PTGp^nl1rDMO|_~O!zna7xNJUm+Ni9D+HK-3|%J4N?`@PTxKr8YGJT6;VtU$|=GZ_-mCenxkce@`Kb-5(1lY87X$
?LgE&reZrO?0~1!)vOIq0wIu`5?P`e1i_`SYZ-6s(cX1T;zEH1%E<J$=U4*pmb}xoY|{t%8D(iRa=|z&doS)}WH>&Jx=OV+*kGV+#r=4WC*Fte(I{ndgy6M-
aD6rcUz8tOZYfp5#=AnEwy{`z9@H8COXmrxTlR;#N29<!miKdqS{+JuPhv)2N1|QqFrgTN;AX^RI#?BfiP;33r&@{FjeKNAa>ENg_tU({Iqh)0H|F1086TJ`
&YpG`@<QKHCdS9B7U&bY)E#{)0B!j%g&aAo(aW*@%l?-)I$48Be+MMvV0>4vmaIBx6!$&9eZUWGgR1iDqlR!u9UZvgrH8*)?uQ)bjE14h<d@v|4CtCn$;JLP
#OnCWy^JaksC0{6aok9T$J@sVtwV9>I@132kdHHjclGqFZo3iut$1^<OBz1<KH6@w@W7MwwCDHUN8?M~#jWJe4tU3YRyb%f0JhaS6Emhcc-xqtXcX^{N3}Z<
|DFYc(lvT7hk+l?>238kdh3G5ZcuOFUofh>v&`ShdqBY9>RI_tPy7>-(YI+(11ZDh>4$+SP!l&6d1)Gfy`L=@r_`gM{BiV^W4AM0KL4`rNpwO2Jx5gGf(%$a
iClf8Epb<rdc4Fr4*7YpzC=E9Sl#_}vM4AY7Bscz?5{h(=TN}HGFvp@Zyp)Vo{Jx!>?->Dd|`2%|Kqi}I6RbOx<*0lhm8u-U;DbF;1}93>pqx;ccy)A<4;2I
IPOZT5~m-G(-bAx7|oG>;@fa#KoFP~MUS7}Ck@eOr;fNghC|1u`U`(S6lnU)Qz+G<;rO?dj7TLI?0W{968pn&Gx}a}V0tQWzu($A!WH{P^j{S3(Zz|_XR29e
G|>5!!*r*>247Y}IG(X5A@yxx%Yttqd<u)szw$K3!W$>Mj+qz){m5cDRo({Ul0^Xv6AzFYxTp&Jj>Pii!Ol~n1E6}CuHb(n3T5dlc4CJdVDI0j@LetvdK`;G
F(-X-mQ0B0d0jU07lPmAwDJ9y-}jC$5S#4DfzMv4;V(7eIC#6}*7581cpR4IELv^}^DusO7FBor^S``Lx-1I1G@RSHdxBwCSiND%<c6A4p+~DJqOoG&ZN#ON
2#FB&Lwt6H_`&I&<9$gFwe&w^mn!{n-}2Jm_y$6~s1Sc66^_AL@X4yMmysxb^_J)(!9&@vOnUjuJAmHRn^9WW3&wkDPMj<w_|*$v&!~D`Fl@5iEnu+0sz2AF
5QQ$l@JzPphbqkG{lD0FJ3}YWef7L)AXfJGtuLlpp!xG=u>T%GEI;%0;T26nlt0}i-E>xToWAYeChLs{Wh^!7y%}I{{vgCZ5(*tX)%Nv-N?d<Ge}PNT06%CZ
S|s<`Lxz`y?Fn3=P8p~9m%!nNYJ*ZKPTfY`w}zQ5{%8n9-kEN+x5R4!@$<JE-SB2D#yw|QAFJ#V)6){tkSIJRd+@Oa_VG828rS&Z^J1Ooik~qKcd%NgFgrn!
Lg!HwTMV*zZyw~?%0|bu!;x<~_P85P*Dpx#g$rh$bEV`KVEl2dI_2p_%<|3CHjWm-kJ&M7$3_eHqOLr+9wm#O_O+P>TWPQf^Z)#D%LuXYD^}W1qT%^qZRv2Z
EyPc^U#$+5#fc}J3_@|{AbF{~wZFj@mBIpKd=XcWlWj&fTA+)64IH9#W0u&eWuDry4@PUkr^AP6Z7}zKm@GZc33PHp7vqiM@azifOnG`6R6gqtCv90mkL6r-
!IBjMG`GE-4_SgV=~(Whg+FeWb<#?;HRA}4#g3S6Aoes^c@L&|A@69FV^!=a4CZMm*=pKCp8M6SkhdD(6z}!+ejR|$B9ELgUuI|-S9Vl%_=B~4v~hG+1P!y>
L=ME(aA=OFKA0j!xSm@q7eAWd_wBKEo^~(LQGAywph-haW05+urz-ZYJua`2&_pcb1!_J|S;W~i9KS-HMOe>c;nfc^vER}#vw%($xmV9R+VT>3$Fh2BE+Y+d
w;q2ym6L+47p8vC7ZkByg=o6g+zHl7Up=%EwQ=;k!SVBlLXj=4aF0&e8O{aNJDFFdaGr{sXl&*lDj&bRs3x2b?#Qzr4IkIyb>xR0sRO<URbOV@bT>ntn{I>L
rX6ZpW@F4;o#A24A>H+`93+LR5tV}(kT0WrMnTA(|LfCf+kN+)s<RNYY?-I=I2PI=cMOKOEnuBu9;C_Yh;Ck{l+Z9o$VtC4&arXFR7VXn$=hs5W@g4Qs?=ij
MJ!A7=VJ7{+-y2u=>Ro7-gm!Z!%*-~cIHf%2HxuOuIoJt1go!Ll{Q@fcvH7uGWnX}PG5@nT%iSK8_HbvS5`yPWt`kpAs@9pq@!aOlria7LZ?HRhg%<rWOoaV
VXDK-E!b#>mY9vJdWS^ec$?;?#0dwGar#h|*{VP)Wpy;6><;EzX1;9jxM9r1@^bQXRS;zqmJ*#ZgN)hD(v~$_aB$Hc9wOxIx|cZd%CZG6I}%N1=>G3LgII3j
R2Q&NB=&gC$-!BYrmrGJ8eh+3zB*=a2<D4*2cPDvBRo@Z<$-1l>^v`>lJhdch;;Bmy^aGesFwHW2^b;#?O0HS_!Yc2-l1@!IFGHjr*)-sHDIK7=3`)@28gv9
m9|)|p!z&>iGto6RYwxHJeaS*rOd<4u}};L4?l7aqBX|u2bYaJ8WTLZeB+m^sRJe-s2!l93rF7P>}PNDOhGeZv!3W`1b-1mQ^_7X2(}-uGr6b<Lz>cu@d9TN
<t?nZ_b<UKH$%vYvQ5y}vgt&6iWaOv0pGhQ<e=w6@;>mOBxLID#z%F9fL!m~v;~z9{`Rvd%wG>hy<-ATnVAP@Jbj5ZHqv1k8c;eZ?uEmlpLJfP#^G(=d&c5E
N4%>NizuPi0PhPalGRBa=oV*cyQ+CWd0=Lt=cpV6Y3V|KG^(Irh(@#9-wp9M8#wvN^q{WF+!a}Ef#VN(FNdCYfI%vAVy?74zS7>Ma3gSbf96R3<!)<u3Gq2R
81X?Ny$}&YW+b-i>KcemU9eAW&a=Nj3;hoxV!m|CVUngSG@ZaX=C|&jIa%+4Bagi*IJfn&PFnhRV%h@H3G%|@cWqEATUBWATN=ULQU#twHdy>~p@OwI7T#HT
Bg*^j(C;R7l(JSAi@JkvUe|k~zwrG2t9{zwx4T-?Pw@NNz25>>O+uiiF@2bx!VaOGzqvK*v{B^M$uXrU2Duq(FJB7>49Mv_GWXfyx3FHWnu7t%Ke<<Q|FXr<
*m!8xya-<8d61o@F~b2tQ$x9@mr=|p+obf92|*UK2j1PYfuzPnr=pMt%!V9eDm5L!X)0*Q;;o6{$Rk7wv4&_?J)O2Bl7-dbz-hH+C7k=#c&}_<J<L7@yiz;q
4a3#|+KLnveA}|W%9ySP-h2Miq5*<<UHoaX<mLsu7dvd-p|6R$FM~@r4yC~>VfW;>PD{|%+Yda=)`!E0($rUHWjy8{(b%`Ch^<(mJMRirF+-hu_*1kt9ue0C
S8o_Xl-v9E=})eBIo>K@L&py;a=t@??<?TO$tJkBUl(RPk1#JA4VSC^xm*(`Q2Z@xzJNpnWzsZyKLiZnbuPf=mGVW9#IQ)#i+g~KqPv-CQV0`4n%B-l3Wub<
&b;bPh4ao_e^Q7nRLbj~?%3uLa)dr2#X}bcA3~S@eRBl+!!9n7I$hL+d3`8-p#X|6+OJ9Hau6aHb~4ak1FoNnP7CPhBk}ax2QoQF@#~+Yy@Z1~E^uq~yfxH<
;{)GH{uoPqIu)^`?PLMvUv$RpA~I;<>g6#y7D3>D2hErnGaOPYqR@FN2+Pq>d;T#msQT`?$ETnLZG)=O|Lxa6yD+`Freq6IS$ebU7G^kJc#UXwuNI<xEw^HL
UE!6SP?P8-jI?8)l_74AJ6_yG$>~XWORYikeNh8S4>NTR_Sz$|QLDCx&j%gDkL`=5OrUiC!#@{6HGH>ye@lkyz<>F@*74h-`<yFi1o?V`-U_2x!)(>5M;68A
16OYhhJxOoVXonaF(&K3yIyKn#>RN%77?`}dSc&ui|;C6ZP@OjEoC66*F*Q+=#Rz1;ikUsdV4Gk$E+rKJE0<6>3R>FFOJchK00U^hx+=z#4&`rKlq@3`(dv?
#&zGH?b#m!euZ|UlcL3t>zZAQEOCO(x92SZyOuDymR%{sY=TRlqB=e##2|4v<J+2s4a%~Tc+=p6*xQYaVOAcHa!Z&{m@YzAi^#PnC7wv)NSzA!tpOX&A)WVh
hA6vC`GwV30wKdXuR4yq;_E_;YCS~(e!sZ!(Pk$FKWU$iR{ZtGuebuQNC8XGai4Se-+O|Zd$@!*0$h-h`RhSKtPO}R=Z)wvDS(U9;bO-dM;t5-G2t_C!z;Cm
ZH?Nx2s=P-!cb=e*W;rDSpz|!OA-EBc)$oVjqF`NXKi76YIIr5#TsY02KbmMG~q)#C>+b4il&>3q8CL8{S$8f`PL-@FL)KLy^ivSbuE)V`G6(Z9Y;m(?Bqfw
{OHh?x^(EcR9$;s+JX_uk^|*=1DIg;Km9Yw5gj)tBNG;#AkY!=or5Y2a?{^q@-{R;`H<nVHiHKumdO59Ub4p<s^TX4HV-^?|1j3`(h?5ea`QE^jKFm2&(}6C
D|m2P(Fm>tpu3s>tCF=L;%-F#bI|WXw)?H=trJf8<iA0zpkxNhKS%OHubP6wtkcVS*B1Q8Ukj=ow!`qkTZbb_mQb3VGn9`sgJ`PqhtIM1(YMR;&*PsTs`#Dv
*?x3_m_qc?f*Bjw`{cB?d8A@}Gx?xDA<sp>F{y@r)`0FB_nt#7;n?A<Jx!PGgv6UfPnRW(K^LCh!Kq{k9;1KSf^3#}*2MB>;&&)A*ImAgD>{K&sh&Np)&z#v
dmR}>jbUAOTO&kL18XX)zA{yQa56eOQWoe8o5sN8pZVIjp5`^0G8l#Flq;$J$<Hwq$FXym(6`>Rad{klx)?qG&0Ig%0vBy^Hz^7|!Fj*6>qgiW%rB(|jw{)t
F!crnnYAu#=PMHpBBC*PB=LHmmluW(6zS52nZd8HaW2<A8G>r3cS{Hyt3kIbrk)avq6c=e(gUe@7Tp;1O27k!tFIpWtp-DQXC<uvmgRqa`p8Dc^>l1FDDsyU
qa}Q>BWSc;`Pu_qIqT_qoKd)=TUak{rjPvKPqYS|5im1*JxVs}h*8=&VS|`x91Olzs2CrKTE;Z{8U;r@ub=7ARqnuoF)jHsYE8`DuI{P*qmP?L)PCFN{h>*J
*M=<B4!pES7--myv9wkHsAseomE<qYdcv9zcf4Ijtx^RX+rt}|`fM<AhVCxe4M(inGu<ZVvx9@$?5&h6!u<$f`jcNvn5W;uUas8M!TplGo6id@G25h9b;B_M
Ble_}|Jx@oBJ8fxX;yOx>o6YP4);Ov?<8tTVGSHtOR`f{wFAw<qja8KXGjLWONgdPMt*1V(9)7As!x9TvLWV-!m8}Bj0FoEl7DfjI!g|<W@Nm-#T`M?>oqU%
+8t5>QjZ0#ETR3UCa!!g02LL@3(k{H@S&KWUFXpNP0~1#^|C(%^Im>h86(W)`iPLM3U7o-%xEiL^TXr$w)(5a*06bJ|9G^-8XS?T#p6?UP-O|u&kS;b#Cs<`
>RVpeGs#o`c`gv}Ze*^=3k3P%4|}F%GrUxZXdUk}L+bd@wZ%9+Y{qpPG=`a=mpW(V-7#AbAM%{0oN@zg%^5k3XB7xy8jRtl^n{?(s5$jxY5Y3!J2mr%8&01J
`Sg0t7w@dJMmA0uf>Kyv@M?oSqGO4|4`1~`IRB3us(*?=O~=F6mEemmDP7(Rk=|%#N_@H3%MROZj1SEU2=^o6WO?rePw*?3mpq8j#Q5o&zDo%f;PKtutJC2F
rE1z|o=GlfzRvlv*1!^8vggMc*s`GVhiQ&9Ob_B?=HdE>O)*w-vUNhs14dK+oVNU#n30Jw?Fsb3@UXMiJf|PzTuWJ=H2Q+d@;F;gktUubEnRu`!3t?vRMSGz
mS8L07>zFSz%C7+#XUJ2I6W(0pU}3(wPl(8c@~!7RQpag{lOHekF9<Di{x;?hp%=f!xi&W?JkU;JfU(l?f6n?8qSP6)UOsh;@H^_9HC7nXbQG`eS+YryzZ@3
sx1B}__#;1U|A1E&#PZ{mnmVHZRwl98Cy{7<9-_Q(FS}SG)`P`SqSgW^H8`Mjj!>7VaYYJ_|pE+Z}o>AdXAUTQ~4MobE@Bs?uH~h10^kA8T%qfM?b@u)B}vP
=>n&1qi~;1D&*;o0lWoR!g>2$aC~Ix<T0gKe7$#fD3ab4(%v4WDQS_gvpGD*eWe;^g+rcq8a(i1B<jaqjX+3*eol|RDGZ6(zu6))eCVbM{dnu91;ieye~zy9
K&z))74K&Yd|3Q$qf6(5`=t)#cLy}!z!G4gXHM{m*1cR23#KSM^P_f`Fpr3nci4_LMdDYK2+KgWBh2>vn&>)Ufpa-6JFJD?cvM_3xv1y{D|>3nhT1F`B=(R6
Hl2sX-mHUvdNpypK>SOEW&rl;dqm0};RH{{^f}SLt~jPD?O@1k0}pQgl%G)=a7@!0IR4ZL5>Xy~`)760UtGs&8)pk`ib<jV9BT+C>}ana_rklxgP#)PED_1m
?je=y42P7F8wPZ?c#=G7S?T2f>I3(>n)8iOaAn)*#R@zA$)9=2Fz$s7r-i$|gt@H2JVBh1X$-5{pck*5BGHafGKL5__)<Aj(KlOTugUL-A<<|I%-=eHSw$YN
yR)xRFcI>%UE}9HnFLg3+UJb6IYUv4`r*bQC%m|ksJ?Bai~FQk%`cEzBAkxPKWNAh>n{y*p3OPpPon*#JHh*=YJ^?%$-Pl5bF^ETUlH$R+Ag_hTjIg3%&35q
MhHGJFlVr7h5g>d{@TZUv3|i};G8(`fB8-RcR$&nf<NpdK6|(l>MHj&)wKa*EfhX8qf>RV0p|s}_aCE7@vl8V+_YX7ts}-2g=R@8*t(azIq3!))0O9~L-wGe
RFnM>^bi#8e$PW>ts!kv@<38M1P&ILJvx^>@#}Qfsg?yRl=B!Fe*WwSV}Ap2Bb{id-0=_*-xG$L%nugoPFCRAx9vdU|2@BoIKcHm-xg;ZsKbj3D<Jx$QsvoW
8}u0&pL*kJg&mVTrK!IbXj2x{%lPRHE>jA5&GRAH@N}$-WbwiJT0zY>Dodo6(iwL7FyU~CY^s`?4)_k4-!960itgppA)VtkXdSwK;*fC^6uX_+eJI4i$`{E?
|Hv27iV7M{sh&s+uBR~Ft;BTB!fmcMR#+%~q{&5Vj_-yFG!Odi(e29_K7Bh91{0JC79XM!|5WwVJtYG)7Nn62vWkJXj3eoKzdtz4Nvs{|9MD#%btG`t2Y(#)
Ze0(OK*I05wc*MpQ1D&Zcs*x?6}^7$qv0M<yED|?=g^OO_xWDcj3;<Od0zcsyb}1MXgtq_*@4~YsD<hgdr%a-pU}231=BmND)~K9=o)3K5xW-&QuX}zhg1?F
x?`Mg>ZgIfP7!9|9)x;WXLsHV@`Q}vZxi<6W(;J<uWCN_#07J;UJ6Qgm|qr)kJ^d^Q+{gy3VkGQpBJ0rkc|M7M*Pi|25$(37trT+Xkz~9i3%2HA<R2k&Siax
M4n60;PBBtRGlMvJbOtWfuC+q3Hoc|tHk*z_A+k>&yx&?3;Kf1jVJR#mkqd>Gm1`y^TP4MB?`lgaJb##D2Yx?MuU>atlbeG*hVcH@9U5PUyuF?`ZtlV?D|3~
Qeh0&o2mPoh@A29nuSxTvO4Y_p<CY{ZG(m-2N(HDKg^RV+zM>;fiq9^B;D%>T%L~5;JW7u;zzEvLD4!0AvRW9^>;#n{tLf}B@L8reYizJX@Kl6M`@)3Y~fp&
*Agyoinze<fA6MP<3)hrcvz`3hIB_$E6rT76t>xX=#vJnG*FORRcS&@bi_+7I0|07(}wA{T)=gA<6~Bl5Q4TmTBU03z!v;wt$Di@FHavY4;Ku=wYyi8K0a%M
*oiprMnZ1%wtw6`PwfPnS0`w16&s=NXQv0_rZw2RUu)Z>I{w$EH?-7>WM^%lw(o1eK(j9lxEu8g&0Dd5B$`{{tpm_`wAoxv8&A|9|2R7k1uY{w^0y4BkQ93U
Jg7<oLXVpQm2POG`x|qZ+HepmuK9}Sw;EvcQT_Y>{jR#M>O0d+WeSdfUq@6U!yz79(>ebq0Gdyhiuw9w5H@<ddgZJoVm2>vEBh)!qI0eBY?uOmC*GQk8V<m7
+OMe!1TWds@J^$v!x3lwg2yU6yfJcF=+|9Bp36&LSlMZofkXVdgwhWS+|+EG*6qK7zQqqyCVIZ`r?zG+Ht|5gzQsyfc}wUUCK&o1wSi=2J>NYhdz^a~)WIr7
;DzVyQ)2}0-6wvhVz$r&4Yzo>xyu5e8+%>-y`v7=Zpw>N)5*Y?_Q59(fwTA^kX+r88x8W>kPJFedC-NJB>2C!$GhMt-}*fsSkMhegPbldve52Gj~L@BWk}{w
pB=chdI~-pnILeblPo|~88RX#o=>QmVD#m3@DW!>NPV5oGolH@0mTk)21jdnsNH0w_!5eO$Wfi{AQ8~86?{>%w}ooKo1L={4dB);5k2t81o82q`fdas-oD!P
faWs6v##4czhG+w4~<(-gUQYCa3M$A<%SaEg5&M~w3uOlj!SKTQ2~Q8EKD5Dckupw6U`w55sciN%Oa7L$B+Z_?VlyQp#Pon>e(|-c<kKVuQ|qt?T(tnQ%`Jh
zk)qJ$juT_Z;W<+mYCwM^{s(sJ_o$|xuPX9=7}S3?LXZ5Eso|&6`sz$4tVJJQJ`fv3}UKW6wHLr(ypDQ-JO~!>zd8}bHWqH&fhHbtT4ydT<b$8lMHYmx@f69
!4779WUZVpERlTTx6lfmI({)!Z~u+5z~A61=DCxO(95SN5S~$mR^06R6?X#^-D;gTq_=`Qk@(7Lq9^w7#&FV<pTNU8?WZGdqS)2Hv++pW9rwv_hn-9Z#P9R3
N(`O?TTz>~*;E*O75hiD-s^%wjhm4qNDsYDw|;6kxkE);LWQzk0GYI*^nK%!xIn%6*{a$WVQvA@D?ZK;G_VeC?bU{=Tx+gbjRrnYGxI$!<HT&(#qbxi7D%co
7IEaY25F~Cs^e8L47t+0^(S~iILBs;&<RbjB$s9$J}8Ch<uCGk0xWPif}TU4!3q5|M-0AIa>FX;7~2~iORPtmeEubDgc9iq*PGe|e-K$9*=Ev%^Lgojc`^Yk
$N0CVs+z<1>Yp%)^SW>?@Oi<Ut_#}6m7{)A*5G<sB%mcO27C6@bGPI}&`G}hy+~RXq5DXxf6jR$BBXlt)Ga@Bhez;rn_mHGO!wkxDHD{+-_|c%)5X6BXZ<av
O>jqvUtqC~9fuDSD=EEqLXY87g`JZgs5Wkwc?)GcQCRC88Pr53w+qLjo)Q?C1Z>L7t{_iH(!XuR9P+~k?+v&uz@SuXap=n#{K^n7=>8rJAC0#3t7>_mT82kk
o&_FiR_F|*X(2D3GN6f757ZN9>K7lVVKAlopDKkX`n;&~RJ-l5yG~mD;hYKdnK>zTCyemY*{e^dQvhPqw>a)zlfqtV{^4sr)|j)(`7-It4);2)-#kyfap%kH
zhU2OvF1x>a`l21+Hag2k^1C;A(J4r%%|FzpM1f$BIyn?cDtE(i<%(qrM=yzr-_ZotF9h@_;Ge^CG`EQ9760_IRBM-;w%#t8dYPk(3rQLTFj3h0#U!LHzOeT
)lK(8*<G|GH=pv_FoDL{M$AT&B@};~a0NZ~!1(#N_gg!*$g(n8|C6l(f5{kn23j-7lq9+luXv-q%zgir3NHwSM83bNC;wl5Z>IIp7yGoKChhO0?R*9}e-6u^
P0@z9{8#cYM-9}AM~9e8>*8`hY0p%W1r`$>gH#$Vu}n8>BV_A|b0NhvimYBR6IV+jjY|XZzjzZS!e<M;;{*Gxq;YKB`&$8r4jz7ylfJj4gT0@E58CG^gE8;2
2U~I!%6bk)_bXh->IL05Mx8$BXj0j`+h_sDF<DD~WfPo#`cUqgl?4{U2U*_bMnLRcq(JM4A%ROKRSmq_@U~EfHF4b(B8^U3j|6Nm&bnE5XIB%uU-=GJ2nFM7
Vf5nJQzvl8^!Q<$34YjKs+Z{}@N2Qu_4jQNwwNq6-~E!QgG;WjU)=PNKq?2xkqg6~V6iYA6Owd6LHb`~b1oOqxKVzKQxF2fE^Xc1t~$KspsF?$0<P8_#J9FU
vE1g)ZkH9b4m0kHlv6=qvMJ5cbqD+!_%OKMtAp=Xa(<UU6XEp6liw~YLXPM51otHg*nN&UX1HsK>B>gOw?dA1sKCuQ%anvQZ{62F+#Rr8Z1FbunjKCV?@?Ae
WP!r@E;sl6-r$LD->x7@!u?wexnIb0;3vdA`{u13=(95#j8rPH+%9rN@TNN+3Jrzxn)rjCa&P&b&vw|8nsH9h$Onp++Y)VQrjV=sE7N&g9ydHk^KSTi<8~Lv
A=7OmY$j~31rgTa9`WkmP#G)8Z>_!0{bqzZZjrz+C&K#hYk1mTm%;X@!UHdV$G|P1VZx=x027OD_gVykkhewTTUFtNWA;U2Z=1ETn|GRwbIJic-V9$xLbdQ=
TGO*c&k)vLO+i`-&S);MzLQ$3fg95Ae;i~~24ju&?cZN?@X43bx`)*Wh1%_jPlavK5PJO4pK3EOG3siW$7<l($;1C1yQ-ttf8W>S({^BJZY~xecR^S@f8D+W
!hOB_WG?%e5@^qG9G|BGwAJpL&vN;~uS7B+IM)gi9~ONsst7@)BIRq|9xXhW$~qirYl)BBtPA8>gn2m5|FgYE1)4Xa?#t3x;63-pHGe-R#PP`ZWqs8`WA=&Z
f2^(`QT^_`R3C-`nbgx_v@9?wIM%5jrH?tmeD0`=PPi)vIr{59SUtzj%TJ>Ra`OQduNZys^?n{}dfJ9Ff5sJVE=eHD_Mc+~ojxWK2dbpTjsEM?T06lfDvR>?
RHf0g=d%ge6HfHf%o*V6cyrE3h!R*7X*P3aEiv%a<6w%oHQ0U*y5*dZ!%eMKMn)Aq(5N3{FM48*1b-{ax@|LvZ_X?mbScA~>Ss+~ybgkAr$U--95JO~d(ELX
6>n~Ku)QfH^uHpVVNE3;zV}O|Gk9v^?t(*1G=m_@11hIqvbdt;AJw0;3DIz8{1$ZZtvr0Zjy$8iA&116J=>QU715c`UhMq99>mkC3^XY!7^v4&V~`+lLJO0k
)-?ev7P1Ovj%z??ZyIs;gdyC5s`7OwltAM!8{ho+B90Kfa^cssfZLaeh>Mz)c+A)2<zuRYmYGpn{tNbKOPRJ|<B39Lml}ir|DIzRC=EUml>s?>(9PSd7w{+d
`$-!~JG}BW2{<XOf#lD&q~yfzaJWZbd(b!-`zPCOOmE4-$lmGS!)a@zZr!r1x*Y@CbHiqz+cl85hlNL`Q3?m<V{POIC2(%>Lw)6sVwi~A>)rPk0F7fa`$Gc%
yP2@^H@ve#?U#YTo9;Rg`fWaFU0@A2KIR{kqGD)FOxO9-YyolBS6=jyvas#=^6Yql5T4lnBYSSG1+q6*dWD2JwOIX~{auY7!gAiJFbQeF==K@jQY(Jcu(UZc
h!|ksqRzih`us5aeVa^bTpa&i-rBy$$^nY#^*O%@YbdD*x0H);5xDEmp`%@Hc%+}@VHdB9{8QoTR>4MKk`B(;DA9&x`w<e)F9iQTE@&YqCj*UByAvk{6p&kX
_rfhDOT0eHC>F|Rh1`)f#Trdb2v2=@vE=N5Ksol4ri6NX9DIy4?vDtVE(}p0{j7+eFE{^w$}zydgG}Fangh^1yQWuHZ3orN?^;A+dSGx0c$$4g63T<;4m|8q
z_H&$WpNe`xPGA{#795}fl-;q)z^<AP{lZ==;{Te@@>Ci95lk)PM)>ISs7e8?lM~^eGFUEb%$!`ZLqK`%WUFgfQ3KuYOT-Ju>5u|KhTxH(LX-RND%m6=libQ
zG*4ghTMF4?6DHMscb@r+Z13ieX6IeR~;YJRdbKtF@vLN<!_N+XV5mx-0yo{8;)W|*`JPD;q_<pmKnQCczm~??c%f+j(;n+<yy8v7DLleYOoRxi=CbxqmqFz
pGUCvvJUL-G0{4y3nJ(7>=2)#9#pf5^<F~;)%DCXFZZdSLXcWN+|CJ%En4g6GI`(-nzc&w<_-?c-tsF`oWz6s`b^i)mVs1L<1f!?0jSBpv1FYUK~jfd&g2bG
On&vfb%@|8-tw<_@~0KB5oX<PClQ0`tfdpqr>!vglxe!@lQxDevxYCFS-|4(r!J?Vi!iwV+9bq+4Wv5tB<EjhW32n=uRs1);5l$VX6%S2qUNZ4vnv&`^G}?y
{D3MVIdc<_{gy#INxAMW6+ei#pX_}{$baU@l|k|mMtmfnRdAv%LZ8P&SI01M%yJ!c?5DE8-on~C#zMk64Cb3X%aucT-q^=j5nc!nK9~>TQp5q~LoFntI*=E1
eOs670nfSb>`a>N$O;&sH!kEv?(CD`9BV1G?r9YEPtiqMRg<UFy&Z{!^RnCpKa2^--MN}lFNNFRzxP*8T?8w&+fu2Y4tizOw2zI5z}f&T1|K|d@P{nL+O`6`
I!K(z<hiltB4#UZDg#3(@BevE6?aIy^_oas;VrP!Y1*3z_pFCqBGt@j;2N@Ff7SqHLG8rA6UN{k@!(40G{Vd~y6;n$rSM(<pVyq9Axz9E<~RPS!ud&r*zpWa
=%1b3-E{K?E7!dK0}~l^?riMW3bO~r<lCb6ZW90H_fdh1W8?O2m`(aXrX}Zx9W#@&0*8dbXn97*B9|Y=Zzv^ybs6GRy5?o_G#Tg*tZm8X1>j|Xlcii$37$os
4b%G-1_sUOHTo}VP*wMrPCl#*{|vUPLs#`-&o95tMkR(+*MnI<t_z}qxV~29q6Q4_j9xI$&A=XOuhxw_E=ZUyqe|n`2W=kl@zxDxB;Gwu-^d{g7Nd+`<VzQ!
*0Q2YbKMSIw-seF+<XyqU}`P3DjbGVjEX6Qd=MfQ4eFcFfMeEYg(^2?ynXeWBx+s@^^zA0kB^b#-l9ds6x&(&uRl%r@Iwms|7fIKCHQZ~=+}4pQS$ITmz~8>
&xFhC1yj}hUWjRlea|7|0jb}yfy;z?ubMudE9uNk=vS}8@vo<`l)XQsMMeXko!-}67YaoUarOuGNJU7i7LJ{!5Wu_N?nQ&=<-q=O(xd2!4pJX3rl0mT#@_gr
m}XuHIMk~qh&#|?A5m2M*S!~!<7F1`uZ0h4kq7%N{A_V1a4{xzN(6&mUk5|=B+++4;+)w&B^2{VGtX>Y0>4C6*&8kch>?vdD0bU|hVk33jJXgjeq8uq`6?6)
>5f-abTvS9;7#7nuo42<-(G%FOjr-wvHJrm>UhsOP(5{53PI!eE4yWczN)O|E(S~3T$+EA7b=dyOPBkbI0$+8ao9D#UKtPeJ}j^!^koY}BbPF>3}_3!i~65F
L71m^z0R}^LD;O0ZSRN}mMwx``xL7oo@ug}=jb_Hn@=h)>6Jo}CX@W%BT6`Ma$xbpeOY`;XySW7B>*Aq1FSq{2C&U<O%@QB!k;G>O}R@HKz_|;Wbud~o~+a#
9N-rQU24-Ug=|BRK8aS=-H``nvWL=gh7`8esZ@R+6vF*%GdA+uDg+;DlboI~fkaYU|C=6Fv=6#=mb?&y+Ogr;ua1()>_1;Lu%(GgbDI$hX?|FKa&a|&&x+mV
FS?G~{<x9aE~ZOnik`HSX6K`|2_9Rzz%(a<1pfJb^eljbzT=!hv^<s{iN>EwmW1euaKS(k6U6LH8H!oj;KU7mdQ&Y<3}3taXz`R7s0?F8q8P0p92Ka&Vyumy
1(i<Ms|mauAHwL7p$;v{-hvhb6)+9P-lVhDLb3bG%3pd-$aH+0qTY7?|L+BVr!B}`R6}BgOjZ6f1@IcKY1(|3#KCS2wWd&6Nb;$k)$)-;HI4H#kKHS%QLW%O
p~L|;W~bk)K8l!VW=<!0A%)Z8e?D2x%A>l6f}-P&Bxs^cI3p98;ISf^N;JfcL5BzVEi54jHC~9g&LWM=b7QpChMc%n#La3mD}}~!?IgbeZfJe5H$K5Gj-QfZ
;<E>QQC7fLH!!Y<jAg^QoPWYtGkpC%xA`JQmDWNQIBcOJ(t2+FqY|{<@aC>;T||nX_?h?PT+q{BqW64if*Aw8I4NHx932b0+sLjCWeVXq8g~H@y^fv}>y^iO
_4+mS2tRm#J}UFHQ3b&#|9tq9k_E2|uMbYxdtpfG(e6QOap*CJcxmzjVodrx<lLN4T-m!?n_&g#N)_jVAW>-9+FyU7X#yGTQXLJIe(a~E+_hPdf{21nYr&ie
j#u89^VhS1*Ps4l+5@r(P}|;@zJ3Y9`)@Aq_r8pw0jIbojYz!gWN@oHt&MyxwL6ZZvQT;X_#HW~5$KZQ)t1hSBHC=yy+zO%7aH#RQWIT<5c}D<l~`%~IC>!F
^CUki2Q&U;{}9C3w@_V|6azSfFOOvCE8)#gSu+1{DQL9s{aG!ogF6O8I_15r=sBUkGQ_6}I);DspQ-sV|7PiaXpj(I1lUw=+tb2zwNX@TL<reSA73_59>d6F
QvKycE4;f(>YOI4gy0V4=_&;++$KI-8J@3+j*|?dhPGayB+>otf=iH^7+f_Z)5ERiNsS-*x|li7{7k{m0&I#gn@kHbP_2`hbj-6ygJ#o=qy80K^RT$C@ct53
!v86C=@U53qgvU%R|!OO!~-#e<MBw{HykGSh8bNRx%R3KUUrl?>lQPijw_FW?UoqE9Ny5}S+PaDdO}gx3k}#PXa_pnItF*=0jji!^Qa2$@~pY3h_f-)wht?H
;QY#2wVRiUaDE%Ib!Jtx?V6-(7w97T2-mOcyuv8^FyYhMFNu)78Dk;&+{jmY=NolK5*=rGpVZchp+B;<vp`A#h6lpWc|sB9%J-R8>#pFV{GAELh0`E8aNzn&
7Bk2hM_n2-)rGNl$-9AE9&8v>#0On9LE>3bG5T0jJUjY%^Pw`KE@;Ld>4crfL%#FDaU%+ls@q9*E8qaHFBj~Rq#-}}%zu<f9G#cGI8NnDVmqR(E?3v%{}FZG
;XwWW-!B?QMPw)0A$w&$?Y;Njdv79p?^*V!NVZBLAtO;#q>R!+MOK<b-M)X^pWpxIy3V=I`}G`;GqIg&VP4R|g#3->mZ50_Waw#T{H|8S(aKka`#wp)*k4g5
ko^>le%Y}{8SeFz3#8pwU&&xUo}Zi}7DHp(M4?cs0|d>p8=pxfLuQC-%xPW&BN7JplynrpB|QH&?voSJHnL~U3q?SEw#1f+nI12SB}kbPCGclUKmT&8A*PQO
of%OOLTuWJi;5&B$c+6;B{MGzwqWT-1|nvx`>PsnaR}nI@1}21vkBPN-iGcl2;kFup)$2bcQjw+DZVX34mD-jqTkOYAzR!|MJ6SJ?#icbi@(_*yD@LBY)X&e
3ZteaLscBe5JucjD+tCIT%uhQhM4_3afT^7>=t~{O*5i|vGUZCtG^r^pM(~WZQ8@O&Sv^&>wfrktFunOw8V?n<kwNua`63bo2f-@hWpFr&G)M<G1<Lz|B|R8
4iS#j98^%jzPdl3!m0(ZpwKd0`_=%D^fZ01T5v#b^=dttt}@OLxSx02l0-zci?FYsGgRmK9zWfr!AOZwOqYNb*m7*ATy%MG`e>Ban!N+$WIwvA>{kHK<FlW}
N919u&mTzGXMkS0nzk1&Ofb$iOrti<k2Uhy=rJ`hRIq)`6|?a~*ZHlM<N6x^`FE7XO<MOpe`Jw5`Eq6#qAaTY$>U}|)R|EdJ#i5N;mq-<Baij)=k<NLiee4u
UTAv7qvVdIa52Y~=4g-x)at#G3B<O7=1ThmN=$VeP#~SX0Oi9HR(z+~;7sP1z8GwP@%;(qYn90$Px>kp%<ljX;~>&MTE?i`PM3AdibGXj`?dYeUND~~5lAXl
gHPAJ>;G!Qq2CdtPwN4QJvle2aOOPDbhC5|J=Vm+#}T>vufn1F>)&Q-yctMRbT6c<+TavP-gJSb0%}N<B%bc&%Dc8amX<D24C%ioBfd?Jk0cK4oa)ZFNtN{L
{w)U>)U8u+Yq)~9JCTv9(HHyZN6u1g+hb_v)5>e#YtVjo=jPxsQ*hmD9G}d#M|AL{_ztx>=-(7SwetwVqNA;JGj9wIpCm1qfi&)Q`t3XoI1AS$p?$gKKDgJA
?mN@rk8|w4?`LQ$;FmG{!N*<;l+S84j9#chSoHeQ%daC*=kg^~B-IHL&;4>#qNBk`P4r0PmKj!a&;AR@H3a$Z<?h2TLvZUhgB(+h3+~x9?c03p1@Q|kT2(EE
*oYt7VUX59*vsH{(w%NNKC3%Sn5GTlwC#*T`}X>i4T<OLN`1Tu5+ZqaZI4@<i~HCM6(Dd<hGrkJFh0+@bPedsqj*ShzwC}Y&XRrC^=Q;bVC%PQE|aE+C>~?(
>+pti;-K;=s$@v<^UMCJH^+4C_Ky)lJ3Mu%Vmaifh{%Z7CU+}r@#@h~e!*Y>Ru>!^cAx0LA#Z_ja&J4=l+RwHS{Cq*Z44r`3V^bAl#67X1O^9*EdEN%L5@;t
OnFQiLussOuj5o<+GjiaUQHdLZj{f&K6&HS=?y`{ojra2`rfnUXp93rFD_>J>m%`cC|_JmBw{*09b35T4*grVvKZdWfzray!Rn_NYVPi?{fyu16ER}(GAlW_
t<*&FOEw2`{N%<L+0=2)`DBF2razc3Xp$-M#v?4AUy+714`VTW3{Jm!@Uuo!``1Hb&`)VO543p0t;uO8-9;ZP^Y!mvXIQ{XCzV4{+zE`IwvO3nsG%m^H0b_=
P_U)bkUii~M~s7-_5iORnsp!7M_H+1bu@bTi<k-K{@k63YVroD(!DSS4jnwPSP}>>ivq2&*ijB+pZ|1PMfX*)r_>2rtHzoNk7B{57xNy^6!!R!ztr!YJ<J`w
eOC?`KqBv#q0T`QoXOX)`rq?RPvdyh=4oPawQ%>{HB=#7O!Dz}Pi5%qGW#EW5f4Yd<Dtstg0RywZn~mniqQ<pLpvvv!B2LXHYY+JG?!yGcjtw1jGn<)ug3|!
zZXBgcqNAnD=psnTS`c8yUyIQV}&5MwtZtxB48sP%b6Y!0;l|E`cr;PFbk+|5g0axwrknX)OtDC-YajhrPY8O59QQl0WnZH?iz-Yh@zLk_{r<f++fNl&>$rh
$FOWT>0}-`>_>UaS#?A)c>YZ^t*$4S6`7@H?>Hi;O6Y)8gc8i#S+zWO<bW6V#OwWKL9iu9^Ug#Jba~Gfm#XFAFDJ0{LN*fH)Skymj)`J+*TADpPaZEn@^{!P
>7nfn`>|LjPWaCD@P<|?q08+2LBHQ-un)A3QW}tjsp8Zy8?z*&MnB%Aa$?6tos8uBQ>K`&ZZ~PPJ&i_luXkjkatOakK+3r{=Zg`}oITg|v7oMKBBiT}?oW={
Vmbf2-uR(2>47pdEQ!w~j_u`wd)Jmm2Ngy}4TLPqm5@JYQaM^B4MrA&SIc!Y;O}=Ido(GGyR`)2kCZ7f`R8@cw<r3DSGOSO?39Mx-6-<1J5uN-5*^&kJ%f?`
(tbUyPFSvsPy7)lfYFGjw=Tt5AcBwP%=CF>^xXTYvfL$xpBc=_)KT&<gUYJ@v^tK8E`OJ#XT!o)vaoS98^m7*ue3kcf$5ZC7{84H^l9e<EWRi~VEW6msN;@E
I#85-{3<8JxuXN4+_<sAcu5STM%Z?AiF|NN6UxEd62hbbcr}vj)SqsEBRB7U&T^5!zOMl$_HGQ|H}#yYe<BDW56`4UCu_`K6aM#YQyiVzi$9+5u;a@F%e0#Z
4}uI8P1vWopt#bAt8aNBlGdDmHT@h~Ht7nA-s@nb`A_#{A%HP0fi5i80XgAc=Hv(X!7S^h#6GHyr;%UVKW6E{xXsEf>ZKsI4%=KyZ|B6PINN?-Vkx*BP%l05
KmkLRRCUCB(s;VHo@Q}N4RqF2XFp!nhDk!&tH~Z}EMB~`u9qqb+Bwl^#cO~bn<(GQZC;e9%(@s83WBDt`tz+s5!|`{JnFEj8MH22k+fYCz)a7jB?&_ztXPV?
Xvxb#n=QRsl#L^vzdUiI)VLg_;g$RUFsj4rGDGvjet)#a+&lGkMG_f@<~7FOoyMI>&0AM&L{OJCI;3C@Odb4k(fz6?X0EnNHd3;|XtU{?N3Jn^+hzCvc_fb6
)<aB3-f_VG1HONGd=9ELx9;DsK8?%_*PSPFd%AioGT@OYhCc*mBvdrwU{?xkT9-=&KTEm_MTk6(tDMs6_+bc=#hvwNfuqpgXsD?LBkI-(2Ix<zfuT#wD)I1!
tXUO9qF=KQ^2?_~XZQNd@4B<!Gz3IpbH)7K>2nVFrR8&enM?_Wb}Y@ZPIB0Ie?<Ku6E|ji=ays=<k5eno3K<x4(BH#7@0`TAXDvllUB+Yj@nwsR3b&-CVlAa
J{bXoEWLTw7XD4PY+0<O_k}h*iA95pHF+Rz@q|n3%3dFJzu|Y2-w2cWR7+v=g4k!}G<g=DIHaR|g&<EDbH+?Yt+~fxB}ZS{d<U?$J^M_uKnTCKJHPQBF#y=y
`+v~{LQwVK@g7nxT)wvWPCiE+(T@kSPZMk7%I=+g$B3m7`tU2wBq0~9e}@0?&@u=2lU}+(Z&uuU7RezQCkK`*r$53*6JyC6F~eOVa5ApCc}&(AhQAH2*r%WV
&%ftx(?71<*2AF8HN~PmzWN~a#+v&E1r~G36^*z=5tUcM6W}O^J7iUCA68jV)Rgda`nnyYNKP|Z_Iu%LJwfAvs~LE3O-*y=P%Z>ohTU53TA*HLQPz!48;O0C
`ht>qph~e^7H6`;AL1Wwzs5xHz%S&y$EqPZ)1sZ*9vk2d$)SaNt4^4^Iq8;`=?l_?*8EQ{3Rsx%)U<pcfCs}jiQkZFqL^wn<V~_U-aF;pojS%3|Ja-cc`HqH
F!M*>8Ir}#xxfs<2!F7bJvx8=2NfDkvb~&`Vo};f{E4jWuPiZZk6Aal?q1Fu4mq$Hggdv-G(~4=pyi1Lbtsn=O1aj1rS`h_GhtD2k5yGXXP%ttXpjTV6X&1e
r%W+G+B79iNCOqsfO^#}ebCG^-zZFxL(}_*@1^bnCn|={*6kTUoZZ8@8BI<^Jm(MI|3epjTmFxh8<j!br18@4nk+^u6V6S~aziY6w)TFP6KMR+()O>j;6k8`
Q0AgBwx-+t>|qJ7>qK$MB}!<%^N*~cp8%1fO=CwVF5tG7{i|;;04ABmYE>yiBwycjn_c=RyMAl3sL@3ZrP<DNBCJxVeVCnmf*5cOUCDmfqmK8vi*dJxRG~O$
^v`Ks64?jOdIadHgVj6D!Q!+&P8W?0lmAr30)tF&@GpCO|6uob{38p7WUcP&iW?)B+4J;GDLMT8Ig(=-XMoAzt$v#oQ81di+VcGs!P3$fAvvWzpVM2*3`rG3
j7PW&-9<rMSUoe_bW#zszlUnWLUlnHsQc-(^ARMS8vWVFrh$uJiH%sMPovEJm*shR1HAiN@#Oq_5{QSXzbd1b!1AczzBfnC;?2r0m1o|1n5`5!a`!b4EKiIO
3Kc6LgX|s4{vT4Hq;tJ;|EWF%AKdC9b3F+s{!10wwfYFZY@l;vP#dHZRf5*)7I-RsbxZWRH!}Nc-L5pTz^j^YYf8%-+Y}S&jBibGOrxY~;f4eJtw;NOyLfSL
u0TV_To?zmh94aFID?P>a(M)@mGD|%^w8K|t~;%fgvyZ|2caoPdS7@ECVb{651!G%+WB=>J0dX*=W2%14;sMgw%?I9Uu}FU3sP~p;E21Q?Y_M@AdMxim!6cD
1E3Wlmj0;E2rlUxBcTHS_g?T#?Z>*tFZ`fEJX&A(davJour1b)P)Cp9+8q^PaYV@u8K@^qLF4?lwiah;JpV)+=5bO7pKQ3yAHDGcdt=2>%QZ(lKJz)|=aw`i
4W{ckHWaZVvk^iPPK{TcYgNtLswl2nG!t`G1HYN_F>$TE+&MNTkf5W8B0>R%h-Y#5vv}!mvhZ!_H-@W4vz&wo`&7sjy9a{3?vI6AT7x9``h$bN4PbLAZh$#Y
4B16(E#e#+NIQ6Vknfo>zFLY-z2kL-Nw3mN=jAq#yp?_HeL?~Xis912H_c%boTwOYYK7|LkhH4?LijC_len^%PyG7wD?S;Lm=2lzm1gdb-=E){4Sr_=<<egA
LRonzd=r;lWi-RZ;g_EE=Pl6fnB^2hor3O-FJa}h=6H9zWz0_59uw!HXFY1{L0GpQAHFC9;eC8|ojLMYEj(pmd|3qC>CH1uo62a`Y9buJX^+hk?Q%WCeB4U7
`wDyf%=y%uKr>1Nt?RR{M8)b*P>O17Xm-c3pUn3q=lvk!aegv!EEk3>oNTZDcb_jPd-i>dJKhOrk(|Ec0GI0K2a|UVVUfoZQNBZoZqDBd2kD&=YG(Fzo52o3
->m5L4;;sCz$Tg0Pjy&6qPs?=D1=W^)LY~?HNlxs`;qQWC*lNM+b*`m;NC^%lceJ9urSQr*W|zFzf^BtYzqz}KCLseo+1!h$@x!;$Lz4|`m0>xmL&eI<S|PK
nd2U%73XIW4Q$IkOS}1+1(pZio+|0tTmN&3{NuN*5jaJyV9cTga%HFDu}B9PkHv^<mnwi?#7*dexC9Ol$*250?t~ceHm+pXDA?tl5fz>Hf@okB-5riRr1aW%
ei$)@fU}Rntdj-yiuHJty$a}S7<;Z)8sT~YZMl%w9uLJ{$auhO2)A?YwnN8mLd#C#C>v)W-foGrOCC2uqwURD&pH!mMMo{m=G?;ZrqfBYiw3xq@|7{j+#Z<A
vv2lJgvrwv6$L+Sz$SB<r|)nsJYJE0eiaakp(Zmu3r9Qn6?7GM6sq7I)4syEI&qvL?y~uH-x#aqksJ@ali>V@uOXk>3=68o2}1K`IL-Ap^29kY1h2q&f1W<F
DL32>?CHa2jXQzKn+iiDVnVyhI*{^BIXtUpgp?1`^Na6pBc70xa?HUFQ}4qi_+tWa{QjjV-4kv&BUv!>^_dAG?yX-gx~+hF@n6D@ZD`}RIT>rjeO=h^77q4!
DI)D>_rYk2{n)9XO%>4N07*v7LK?F+-Z`tdF>V%N^~w#JY7Pl3?4$qSHnTUs_ZM;FFgKo0iiq9ZFh${^vcta)$s;*Sez!MXA4YV(=SSFrLHx-z;<vIP#0Vz8
#_aXQ5VwrEdP_rG>+M`FZ`8p=?}G>Y_o@*2=KjjDB^nR}Pcmk=D&T5%zCX3R6>`dLFMC=u;&dr=d!rR_=AMmzyAL~(W!y@n%h{m*Aic!kFfS5Cd|9~c1@S8@
{N&**E!g&(Is6bYLnKQM8x5r;$ae@|l)Yv}X`|jihmZ<nU;Lmwo2G>6m6QIt9F9o+$bIVXEh!jx)mxYE=f{T#+UKgz%^+}k>4LMg2_DpRZa*8+K~Zo07v(WW
3{?t|sg(*~|J9|ZzMKkp@omR0{e>;cJ}os}j1z*lzSr75iDdly{K4lTi!q#tYJP3I2;d*l@v&%8J?zv}T8s#~Lg>Tq`C%R<2&)hU@KW+3LOXQpOAs#}e2wZG
)G$LWe`395l^V9E#xlY^dGN{C;k_`i4LUEq4PN%+gtYj)j#;T5GS)1@tVe{v!YrpRxBf@=zxSB**w+1Bj8V&B#x~o|2lAi~2@8@!cwhcrQQM;m&8oHMrYP-U
5VP=PbV3Jhc1NX^?A731`gLAlR01>n7Cu=8dp=t|Sei^viH5?7*CO7Q;DO62o1j=^{XN+v{n!@_Gw-{vnfXA|dw}D`4Gk1JE^MB(7=&PIaSo2!AW=7fS0>pJ
VxvrBQjQ52c<vBF!fy+rB_X=8234fJo00!)9Ew-!q}NG?@}T^9lsa<J8<oe@1Z<rh@X23Xf6dep-F?4#Yu&RETHBzbUoVaAQ>`htf2pDNUWnG4r3u`#*B0nI
<Ag-QN1T-xBVZQzI$-FyGk(Xtb@xk@+v8(Lv*({eAT#>j_~71t|9<xlb0@JyyTt20errj*Gx%_MVOkE=Dl*QQ{3h5mOf}EE?2MVI*oU|7nL$&6OGr)I6E2Da
&OMLR!AU-3!_BIX@vSeqQ7=@m@#jE;@B?G0e(@yB%RdE8eGU`m8@{+pp7mjhLv63GyX;ELIzgS2G$5+R3oAy&IK^X&x^uHP+TZe_hL+4CcD)2_1(hF51FS*i
mQc~&;f}}D`9ZOUVmPBoU;04P4-@qQ<xiUQ5c_=oxD2x~e%H`fbWhwu+VzRf#S}-Z?Mn^lKI8$fQ(1p{_k6+i1#kYrV}`i<q-jGv+88+s^Jz@JVYpP$6`eV1
h4<Xw)9=XGB0ed<&5KbO%n?#g76{LwV#FfS?29wn%sZ=n(jDMTGR!{r%ne+TEaVIdf^c-@|9+6s7&|+@a+STt=wYb4>k#Ax|JODu*4#EQ`M~Y+PtX$C=ZyNl
80sKzTC*r+#s_!Uzdb$6qX$N#fsm)q^bq0E@P{Z#6O~5wn!4Hs7=0lhD6;1Rhtf|TqUA6F-&_5~7%dC5o0}-;svCi%<n;mFy?!7PLab4w?*fWdTg}Xeitv`}
PE#>4!funD*ABlntWQ=RPZyRz?s|B&(O!>H9Q*Cm<)^uqTi-gY$^%f9?)<f>N(Ij{FSj3^&;!LO!peI(nowKeVpWWhfK5^D8|`c6h(6nPsI<%xO!Z7&cKSfP
w#-D|f;l{&3^MUnxg%Kq+*X*pKRh|fM~|(#1B(fg6-A-=a?Ru=bBGNRBJPaC*a+4%uZakcIzyWy`gr!9FaCd>PP^07P8#cDRMd%ce9{5G_Z@#XMrMa=c|m-g
x;o(IAhq(TRz-h2>4{EdQ@rfcW-|?tNBQ`sb$Pcwmg*i~<y2Qj<Hz297D+dxhmW`2n$kzfE9*wvVOa=xSR455^*2G<-zuX&+(6+&{nnhz7?;Rb3N%irV4@>H
EV{u2EbK0ZLra=qOM6iIh(QtUqbk2M>4Wh^<EMusV<>*qs3xtQ4MweX;FGcY?%4nP*SST5=O8<ne8_Ra3*C>&UN;2mK>b{BXzw8>j6IImDJ>7jTU8~dFM+9u
Y&SaAL=lTOhr|D{cRJ%^3w30KZ!8qwwiETzro-@ui2m5#_R3=JhYvjRfT19n94+5-=*CMYe!G1hr%O!bSIe`IvqTW9x?qICH_Q6MVy+NPS|;|Ejzo`;rS0#f
VlZr|4ZE#{foRRasnWe0cW>}teoX9z{su|?m9BJ%bu6+33y0%epn_^zO%NVd`5kR~8w^Dm2K_*dXiy}zx{DNgATamR-FSx^pdZsc;$aqx)01@_t$ZN}^IjZK
`<shjNwocJih(F<BOw~C@W*0<?4<^&^Wb&cwf*z&A)dYMn^!c>0sq2l>-C-}jDF~JJv(y;LB49&V|xO?Pfl^<FPk1J$3DIkG<3y{3ymlK@_T@<$6zxv#|GaU
Z>fxAn#1vs;l_A#B3zdPmi<l{K)h~(vWzhke`U<78-B#V{p;O7Li`VrCNk0Ouh<L`Yb{B-v&G=PO+jFwHGzxtbz{qI4p^B^F+WBU3OyUUzH35R_(WCxkgqQZ
q;=m5@@A9w_TPTsPGTgc8CJSqjd_A-am8s{#103!{d+UT<3Q-0-xz$|8RJo}UiLZ+VCu|lO~uFsP>yi&@Ay06ii}4u{o6=XSG<bUJ{pWi$sz4(**<tkk>2O2
s19*UXVhAFB134J%#-Fm=wFsvr_uQ%{C;5RAq^3TeIySwGt7muEi-?BVkAD7n0t#Jbwh@I*Y7OHL?q9q9?G&b#oPcd>AyW+3MxFW^|>et6CcJ46s?2dY+$-u
Pvj00!lw`Z-A=*AMc0zq5lif&bvoO8F$UFVJ^6@+190GAmtLQjGvrvOBdo9VgR>-m{Kaq-+AoTBy9XBF&0ycAhkYOh53sU`3<O~HgYkzh9RvJO3ad!g3I<~h
{YXW81_+I#S*4_mvDU%GH0ZI%|I~3S18?NubKQac(LNu9K3YxIR(Hg_V9i1$g)w+j21TU^#nCA)=B{2O2)$T#CiygTy#0Acm`XJS-QqmqMoY19)gak?Mi+&7
o~(B^ywAZrLnw6SYb;`v3e9f1x`6HI?n}4$K+LU0&l)J|A$?c8KDopWcP{A%M0^WD>~#9c-Cj+s*ZuW=pAZLQLM!XRCkdcE>!B?1R~e+oMt@o#bHUfk+53qd
f<RInVnvxE1V%Yq?l&oWI`($0Yti5Uxy_LC+O8bd-?7f9=&9kDUvI(dVF7$DSmK`IlgIhp63u+d3kWPbYaeSK3j&Sx+J6IbXtL8ey-mJ%{ADe%g{Re#^I)e#
g{A<U)OCbwIUdlDm<(=vej0qqJaz4=L11rJ^{*iIgqP&Pw@~&y-Y1$HE^<l)ADiD+Te%8U8J$gi)@^a)XbnY>M-0?{N4(i}b4O8qv8Q^fFU~d<)R!qHV=d<G
!r?oP$oovka-h`;;e&$QUkTiitZ3QicF_)7jynX`z4Y+HO*TVIV-Q56)Tc%7_`;d!)R}BwJ5Ux~sxnnp1&Pe3jE>R}oVjp}|3{A=yzT$`Jx~io2c<$2+f_TT
XKrwR`|k9ge}||1Ehlz&MY`+Eg>5-g%(kl;n6(7pc9MCQzN;7Vyv!8_Ugv}D*00`wXG3w^Ze7`SAQ?nKq4#CU;;_g&Nx{qPg-;C|gDpA2n3DV_)p|Z~k0)n3
=>GVDJnHnU89^0hP7DuRFepUT+`s<2pX{N;8qGZO%>kiJ9tF~K@n|&@^r(OC42xYi?H{3`IHx#$<BqW<%&(*dO^BO7q<o*o#E1^ov<%{>{7o<`&3)rMlQcp_
orp4H{ScOy;;Ty<jK9b4d0im)#kumHe|h9U_`W#5OgVQ@ctz68y7A(IO%8)YXdto`&iFN3`amT6!7}f75FERn%q*qd!lZr8-#63-7|NQ6^WO-?a&^fQ^1t3F
(IfSJ*cc9aQJ==~aVuyL7&<>i7<O3Lt6x4bMU0i}z$fz*bcr764`I>+LD@IC0}=dKmJlhv%wi4I*$*RKV^`p!JT&R4><lJ3>z!GrG{~!49;eq!K$U3R!A3zZ
m?Y%2oLVyhGl%D$&%WVsQk`4M*7CysFEe6suW513C-CwClQ8HIv+OEO$RT54{p=e@CD<Exv=wKt;a!1SZSK4tY_z9;oUA?%v60iK!|9yC{v|&y|D7pFc7ju8
x7{$*(N@%Q!5bdqlN|anS~z4xW2KW80$j1BbIR~SmuD`F75zX;U6;0U(GXur2?X|wID)u4BJ-S77Wx_54yYN$VV~9k&FLsZ%+xpeB);@U?m`01t5#L;)K6UD
D^5iDlY9l{<3Z4`yrJ7F9g9`L_|+eL&N!YWu<O9jhLrS;arLj^7++?KemCHWsDucK5I#S=xxZg|YmY0~H#Mo%oP#hB-hOj&TnYykD<d)n%(3Mc#??cp3q`6;
HC87}tP7-?WyG1|3!OB7L#ro75BTlsXV~HD=ZE1Qqb_i5&10ve3V`ND_4O2)TAW;a^ue$(A7&g1_4A~Ad4IO-EAhiZL`*V=&@3ikm0N=%LGco%W|v5+*IaS(
J$=Wzxh063OB$GOYQrR_gqXEj1NS0+e$?sqK$civ|7l5OP(JS>`uMN~JNGv<Rao*7UriJja5@;f?iLi0B~EzjdXnhFs1t6ut<P{UdqVBmP)hH(H$+_zTFl3#
;DlD}%%)>17LOjj^X`((e>#=2vv@bgkcO+Bg+yYyS)i@<5H`?sg!JmOW8ELqAbBPJs(gbHJ|33hXb2BS!QYI`ufiGlp-d_HVyzOAvQ)hh6!{Q!`+2;0zzbeK
KBxx#%tSgzc|(|y0Uixo(5!O1;;l3DlZYyNMDzQtO&WM2IlPjOUDRh!2UK@|hs2_L+SJU0<pRo+Y(^Gd3!o;tPkjBUB~s~J8b+SG!=_EYJN8Ws{JQGwM;Oh(
Po%K_0IeowjcgWvshNXrnW*NYN)UV>aP_lpWa6lRHpi%r1-O|H5%FCPz_Ju`!Ap{0=ms%TicA?`=VfZ;)?UB(B|abX%|!tuzjk*E#oeJen3QzsqXAYQt`3Pc
s)Krv^N(`0Ge%OsO<tvS$M*^4wiTx`aG$#9am^_VjA(zBlk1Fxm21Xc58N>nCHkX!Pp7jIyp62I@=!=RR3Mh8geMd2;ymY!!6z`B{FFBsZ1g&G6MY$AzVMgo
1FtJCUbg(Eq_2bq3Z@65f`Optd-jo;#1AZF15S4T+!229$iAJ7LL9L74cA+Bz>IaWHodk#4yzoFojvCZR))?utiC}QKfi3>dyxe@1KLIHbMELLkrfl{aKrg?
9>clWG<aYiTobHg2-$8siaI@AkgxX-DliA2tIBid-%L5GN=A+fro>{dh<dQ!umB%RZ!#)~Uj@;mip-TvH@NQ_eHfYdgZUHYk|PTKpwgYy`x9V?iniA7+V2jK
=KjLwch3l#_dYr~kMUvr_PgD&oBHt54?3t3<BAlr$_Ec$n&U>UXn#d+0957nYuY}Hf&QW2eFR6W@N->-nte6{g29J8C;W1t`h(={?RgjE+ULnj*H*!ss4#L@
+!Th4E{q<0cG!$EH2=f4#|4MB$~}EtAbq2`zV3cB?hwBb84dJ<S@VYruLJLcuECs%<F5}gN?B}<EZgGseThoq^iYg`CwlPA-50$-G*^2F9WflDsk%UIkFu==
uB_R7kSiN)9zE=aSxwud?yww)reAm6e36L5qt{i~_w-l7K-9u`#R2*PPtkJ80JQH}qV0TLAtdoj`{SdV=#FTvdDiL%7j~L|D*IjF#&$Vq7>Y35_H7a5F~jM_
mp#vJDx>69-G|j1EEqW}DWvM}3>pP<@t1d_Q0}&(Uaof$?WU>OB;jF5(p(_z6nDmnDi`AYTorrVCv0J$>HvRkx~o<$d6<l%z4qG88dG;J2$QgiLAyj-a-W7V
E*6*EZ2hf-jA`POG6gl{oyz&ke%cGX@BS{8{=5o5)>pGZTrQZK5x8e)q=;a~BU3ED#Bp_;?78~$KxmbC-v3Eshr^YUOhgVIAQ>0I>d<+FxKn3TU5iBVlT)fX
Nyadoe!ABiFQc%@=hgQcW=M7)Af~J>#N+lt)dR0N@iORs*r8)Vpj~+N{fvw~&NX|r3r$HvB{x!!yuug+mlh_(mCV7Z`Lz8wogS{NO%^^bs7BzS<S!PI#-Q?+
+aa^@fvl!6-|r+RC_X0sG0*Fc7~(&>2TB}4l9WxkGol8ctgqijDoWu&x<8g@!X1N=wk5aO?NL4uq|fCph`#TtN5q6(;VG+BK9OyY)r15=@z?;^>zyIcVhw^Q
S(97eZ7J9#H*FFw`GfXb{8VwN9XdJMPJF+jhpc}zEIcx~Fu$-(7(cFu_Ee`w61R0xtXS9WKID&;a_1w{#({9XL8g}C=!E+pUUr6@I1hMoMNIfAKsv+d;H9z)
`1^*nhWY*lWQ3U8OE73*TV=7tp1>7;vjIVC1VNbXo;yo)&I4sL%q&6g9r5&=!<(iO^8ftXpQko8S;`qz`y7idPwBv?{^9$=cpbcUcrkL@DHrUl-{O>w?Z6}L
+ZB1#6X#Uu9#**PL5E-Wxn7Gjiein9UDL9})K8xz{E0)_`o*Lx93IF^)E6705y!6Y;*rR84YW4ad?KN<!0NDTz2K<h-UKtv)TY?rny7VN#nB`%(O28(@g`$_
PHUq;&=7o5CHp>eHsjXyTQ7D<tU*>cP?K%Zh`U@&s{;I{aBSOs8q?;5Bd^k#ZZR0)g2&v0|6N}q-sqxs*-!#cq=_?oS^#2irT*sZ;6$g(c2LnID`=!B$cN`N
FtHHZv7zRM*vmVQd)ie|s#pJ5Y*`-Oq6{welV&&(SfZ!b(ucA6$-mFj4AI(jR)|4J0$${gPi8!lLHuMwc9B^CK3mlV_07g$+%GZh)QB9|4qVTObL9n7dQ_zA
Y!l?0X<taOs37rDUGL`v10*!hu?A8J!Oe<GJ*`>^x57%j`D!iTHc7eO(q(|>m$eT~a6ZG?+f6lL9jYk$p!?$OdlPWpdAt}>EC-GZUTKyPYm{j4h3ba*<EO*s
->dcU7~htaF6C8$de4^}zL0P<*d$a3e9}b>4Rx)sfHk82TuG=u=ZO#@;SI`RI~1@Qe6fCKiBEZ|H~(1*Lj0-5ZO#RC=!eN@CB`aaY_MSL=4V?lr%3pmiL=7d
s(%Ms+pMA6#{Gf)z8+e9p8rbQTh|;*<!jt$BH(N>v%(l`0+nOeGN)SWL9P@^-(GHs=w6kndqJUivQbLz^4%X7NvOQLH*_%<ur*6Id=9FqlTQEp-hwvKLD$<4
fPGOLruF^GkkGjx`iW2u<+cquR6@$of9$hS*y4`nRl(TmJ+9dBUwKTsWQgzG!qKjG)p1CFp|9at95M&0*a}PSk)|2_*xN=OI~T8PK9<r(`33S&RS9OC8FH?V
=<tHjVVB}766T<2%bYu(B!OxFk_9>wQyl+ZXv}rk15?Hpg7^CLaB1s=L4CU#4*zR0@V#ILgXPcUn;{NZ2S;+Dl^i5L`A(0t?#&@>UhbV915}S3fB&4G4P3qH
HS*$yfSAcOg#|;5`1+i<<Qa=ACQTKBhYfM(<)0n3C0+DTr>TFLbAr0xRl8aPRcMmvy;hWP#Bz~4<>0>m`+nB(hQO!Reah(dwd+Zs(nUgENQ#?;JUVI@4@Wb}
VYK&)rkT1C2D>+E&zqWnK2+Jh@(vKdG(E2S+8nZRZ>*c7?eHXbxkEKnANxMMU*fS-#B-&as~M9bcpB*28TnKXcHE3kqd^I95B>c;Zr2%~PRfO+ma-$^>3hXC
C1Z%OpGwyD69wU6Dtf2h%V^&|e&md$BdFy<o_vYcMP;`|7FWFpbgN$-c%{h)x5{2J0ux945{<rlm{kcn?TtiE@rvjq*ZOuf@EnM1S?9D8B+(ngL7WmSf!l`6
7J*x`h~M1`INHLBEX9Df*JdIp3Q^Y9^7MgZT9)he5esNtUHp-tp}prX)5-+Kdp#{s#Dca^4*QAMDNbc5;?Ama;l5}g{P;^_*UWwubH0`FQ%9t5Bf`wn>5V=t
9&62L-?W3(<wMm^(}Zw#W3~Fyq7o!p3S3^xnt{SWJ@L}81_-7%a*zIy!5zW|i?=eO$jM9=IUC{#XWgfQ_Y?WBb2F^jML`wwU3m_zQVyVDpzi1vQb5_0#<+P?
Q<#0&$<NI>i-0`m#Lv(6=Gvs0-l~lYbHe-1U3$t2>h;_1Cg}<|TlcN-#g-N(v)Iz_WU?bNV!G|CnhewrI&gb@Wk=}mM~1EQ)|ib7IHX=B47P)0`a5oWy7&`S
>qW_mq=(jCR2R+RChy{!q9BO}Em_~<==Snxx2YgZL=XRbY}=ScBr$(Ag{HPk9ZzojYg>D!g|^J246+}E&?hZ(_f(D$^ppAxo@SdPmg6Ne<x@KZzvZGZx~GSp
_}iRMdOZ=hH(a%;VxaSQQ~GC435<qDW~KkaQE0l^xIfhhi*}wi3`v^c_nn}N9JfH^!lTH5Sv?#TC;dIbcLs~fA)3CU!kD|n;pRKw0j@(YY|90dabTi`<*7ak
bQjmh>MFz$c=|>OFS9HTN45%V8}h>Z@cMBYVFA#nA2sJpGlr|wz!tZwCeFm3Xeaxj3F%S;S&85B7|)Uq7<{Y<zcS&7WBrO~8qCn5>K4V>h#;xA;bQQlAaA1J
XM}h)2=y#=a3}oGs=!fh+>H?b7xcsxn)C5r(?!kE+0CUb93Tx3pE14MJ_Q`;nOpoZq=>gkynHe*)gk$SW+da3IFwKJ6^VICAe?|#@YYvu<W>rnwd|_kvd61{
qvzzISo?sTJ;ns}J`P96jhwOCKu<nOErLvz8^vOG^&l!g>A<sP1;!SSs=5t%7<?bdOW->J3;m2rp?9>X2=~`DRd0ndf$EE-WLcculB=O}5P(2-|2;KQ2k=aX
&Ryd2K%=X+i`_9EM0n9%E^pEVgJMo#0XG-2bQk#rjZQ=M{%m~NM?PFXe&=DNpa>SkZ#A{sT0xyl<l-43Iiyz2m|@oh`H$90>T{TJ&g91egC`H%dgN~j+qvRE
>q7S9AOZ*|>|6D}5D8m#&r$Y1Nqna1^zn_815fcACNFtqJh?ZVIiRQo%g?d_RhI;jpZh1eok<WmNv#$I=hPv%f6@AWgC#B;o&T`A;eaghE~5_wT)3*L<R42S
kIC4gFg5c%y+sbTJ!4*y?ecu4^s?0iGhRJcC*~v|)AX9~`UwMw%e7P#?De~=B(JNN=JX(CQKa9~=z!YemS%4biG$p<&!vi(8T0(Rp(2FJa3=kDq&L$Z$4ZjQ
EU!L+8|!c4n`PDz{*<O7UCM@M6CVmjWlUlEBBNdTrxLDlDH+w%si1{PiZ_!)5Iv==zjpU}3q_0R0ZSt#C}|A)qz=ks;_KjM=ZHG4sj||qUa-a8YQ5tW^@b?D
`0T}Y|NlLo#XHyVez(pay3^kRwMsnjx%yQ_E;|SM-uHP5P>O*5m)<KaA#EHLj%%co<io|7wB@cbC$Lr^`R|_y%*kH9NjBhvmsejNYzo$Z=9xFq3MGl)IbJUE
@L>u*E;FKl(*;YCb^1*#u{hlvMv-crfCYNiay7dk5PyH_xb(mObY5|!Ses46uFzIu&{JDnXWH7Y8k&One;#Bvg)YO)E}7+#cRS{w8K1!!2V;r(jj#O|;5{OL
SlBKR*98+5Lv{0TKz^+|MK2C7a!=V~(gTMAU>)Tf2<DSAg)8qIKoq~fQSkgFTx6(j<l)N0w&IOH@%6FzYgLj%x8jP@;ol8xjonZ`RpxWBC=wyISr!FfEAZvh
HA9{K!64}D{<eR^4%!1-hjVHD(JMM9ZLyJqSGqTAoce2!@l&(7s3Z_88G3KtE;%Aqe`k>`stQcHTjXZjL7<vERVnOJ1$O1A+&&>|P-~<UbTow^HM8rWz*|Q+
=!Fhskf!2Cr0!?4)@Xb;YpOQMj>f9Wj==La6|6ZE(aVH{;*8_QVCO;rg1yy`p1bOabkUEOo<CQ?L~-pNIK;zt(|!0x-DM~{>0f8w>%CO*0-gRh<Di~P_*82-
5DxnfK5)2Lg3xCAN5$~Nz*>L6eY#4#9PeF_uS~@A(TnxIPI(YqYD-_1j77dThv@U?G59<X`#zsI9vX9w%k`srDD++8pG(O>>v=CPk}rAa?JJ%481aCXLLV7J
moMto?aKy7YH&7-;dK9rLI{hrnFRzFK&y(Y@#2&!F0R~4uwK~f^G^3D$=j3R_nUX^$YKLd)P?P%%{0W<+S<-d^RPW0v_5Yg?}iTBb_tDRp_t!RDey3^!X^39
6kF9`RM|Xs{j_)mOwyU}pIQ{+=9=O4>jSY+BV@8?$WB75k;1fVM=E}6@=)+(#-shLyv1OtH}*?yt!?ucgNdE+=0ENjsBr&E{IxenQ`e-rh6JN<B3eA;(qaVi
M5#<nt^ASOx5M9eFA{H+xT16{rm;>@`FQAGK9ZC<ihh>5fX(vH`@wE=2z{<Q8Ld`^PGz~7{?FH-z1{!EbvGA@2RvA_F5Si--zA|VtH}tzZO+VhFBeq4-{yB{
_j0G{E9FzcXbf}olUc-+{ioC9?9?oZ!3e~Wv{v`pL_wsbNOgn%JPL|uMa%#8BgDC2fZFgG2rKVZ@|0$yULyAs`yLmH9Tbw4qz}Q^a6{ZmP#!XpSq^k|S77Xz
vu=sB6B-i-;@?qBLhEu*+bvpq#1in4ZvVQ3VAenRGS_SI=gPsh+zvmKXr)F|I)#HVY5AdvPyq63aypHB^pT=(cxdLDHw4c35Z-<qhc50__6O>zSWl`*QT!DS
<3l%cn)KWe$u-`rNN)=No$4z*k$zaI42T{cGR9EoN?bQ%5*82qtYF=WfeiJupioHw*g|X~8jQmsf4(?Zm(K$Rj=vj*zj-4}_+)sHP6#Ho+#?+1L-*FBkh6Nw
37*mPzu0S>q2-sfzf4^dbK&vc6Q%OlriteldhLspK?y5e3Nyr+NHlyRjsZn8XM*YBVtk$O%5fI)Mdo<r$YUN4EYJ=ZdOrw25&I8w29^{M&8N7lPFumHqiZNI
&JTySb7T@_ozQRcv#OIV02AMoZSGA6BUU1@*u^j&F+(mJNrqbZ>#OBQN*s$(`t8?;!y}=kJjLe!<^ovsY#J9U+~H1ZRcs<Fg~7Wl272LcNS#hQ`GI(EPIs;6
nH>cnR<y+T&OZRt#e-}a*Nh-A$*rHjS%cLxrj5<Ftnh}1yDy0*5Y7Rgx!Z4><9BwlOv>LZSVle(|Es10P3Jka5ye3Cc%J`O{5=*j>OVyZPWs{eM&!u~*Yl8+
y%FEP>V*_n7DKTiHH=w(bNN8#fbGFAXOF*0gA;k~8RCL4WXm}Sgu9tz+Ihb1GD{3<RHK|sc@p4KlT4;^!4<hR|K5ISGle+q!BfsEF^HJtIbW-4f*jA??>o6Z
7&-FOl)E_<Eq|kkiyr&K&X+QaqSp<bUIaB<<#xz>eLp2fBM)PLwwDWuC2@?v$Y;mh7Ke#bb7HM>@Y8#UMKwMIoaI?k+F>I2XVuO&Nf3j5+zQi)sTZKONwccQ
Y>z6Y?3lQ!SX^|jqw$0`{QA#Urcqks>W`B-%0iwvkg7DJDv*KuvC2*tzPn-beI2RWA1Ao(|EoGVVF{}ehZ>@@e)#wIXz^d4ICOfH9&VGh#FkT<gv+-*|DHdn
C)4AMXQU3NCm9Vv%ImBY=n{kBuF|w=wtR58zO&SeQO0cPR<{9v3<92?c3A99hb@`C{c?W}29NLuX6~IYj_Tsu?K@T|x;i)7w&z3XEf0K7oHM{IH5I+eQUm&A
RGaF@UtyWYo!L&!4xLpeW^Y_9#NhMN{wybFoDj(|-9DRx9~$?if91PDjn<omy~YS5@eG@K&qDC9cH`MKJ1-0<=3S+{69vQLPG|4U%i*`l6%ww}WO&s`__*g@
2OXOdb$D(NN_4ZMk4U-V?pz5=)vCi@Z@NIO&ZZ2zt>3H5ZK3#jY21<Tc?v2BhS>cU6_6EY@@3zn6s(8$lQJ3ELpn9WHoo90>PzotKQd2)&`E-rwQwuUZrN>q
`;w1)AE(|2YsDb`#H=c5OE8-H@(-D&rGg~z*uK*GAmsl^66Insz`@{C$0r#b5dMkZ-z=d6BLq%@lX^NhICh!%5oZ#p8G5Mp>&D@7e^K7#vkVXlSHxD@Md6{)
O__t)`6#y#;|p7>fbsZl+E+Ozv`$3GvBhaa?8YoNS)(`@Xw%cB$6Qgi5}~5t;)$-GM_%(>s)FIe8uH-L7zAbnIPZoxVnk=yr}TsyWcrum8Yo@Bb@*h!$?y<p
#B^s#K1l^7Y0@$CAIZ>dc&a|r?u6}S-6LT$`8a=VuIKxi0645%T(SC@gW~f#wY)X_|M~ZOMRsus&SXT#PRiQ)#o^TC8DIY1V2Df>klp%df@1DNu0J%a5Og5N
U{Ew1tPVnNQ-)&Dv7NeIvF(FGA9`xrPwp@ZrnvU-%pOmPTwyXM^TyGCktX(*cc2y?w^q~`3g3wUs?Rd+kS>ev%eY~Op9hm9<&Q?e!EDwcw=WHA4ALjlBm=<r
NAXA&*$|4@_FW`Mb_c1W>BhlnQ;1%SJHD9cg_@__k^vDpaQ!Hf;Obxpx<C%$M^DYLH2>qV{k<^c$H2?oAP9C0GWM<G@?d->vwOWjAMZ`xydE5v2GLQU+Ac<0
(5Cj@*jl!SS?OXW?E^RP{`^A7pQ#7e4%I*oIY$VZ$z~^f_r*kewTasY3*6&$qqlL_!3KSza@UjsB;u)k9O*Oh+=Ft$?Ue_rQ@n@lMPuRJu;hK6@(TLOs|}gH
J0Kw>e!SDh6+M@a7`yy-#&J{sqqJ|Lq1(OjcC}dqN}9z&pX|KheB<Cem8}U5wN8D_W{pR6_{WGJpFHtX<ar9o`EUp}$a#MB<;37^nZ2<H#-T`_CMg*U$lNch
X?SH13$aiuiXa)>bBI1#-jxUzO+86fLtos<Hm<aM5(%<Chf=dDJJj;@-EO#SfmP;C-619SJudkn6W;BCs*%+b%PB61S4cU3<CPPlH^j|lNkie1c||~o)(-2Z
YCZ@kxnVF>c6H>WF%Ar;=QEZl<K>LGYSeXGu>1ca?lBL*;`z<1t?m)vHCq{aF`@_JgLyAsklVqn?k#6Gqbs_yxK`99+%Q34sv2>|0)j8hsKrYCkenN8tXX3R
*FRFrZ(N-aG;-vTwW9<EEpPMZ-nYc<SH+#569e!(d;hTL6$^Zz_-4VtrVqL)@%fHUV^|B)9K8SA5gpdoyT$qJ5O}t-^OJ}X-cb^L7PNK%nOWGYOodcr8ycm&
7<PgDt8(^5OGkjW^TYkU913Kpx<Zj=4~a$b!w>kJ5oWKqsiW(FPo_6t(}vi?!D(tVS=In?cI-iea|XD+DR;ji*#ML0_&w;geev8}BlU^AHNuX5(K;1tkHnkn
{Uf1HP{_9zK2GU~eX5Qx<eU6(HLAsXz0CzDOiFuyoKL{yp(7;<t$`T&W-n&2YYN#OjR#Bdwy?S}d*pP5+5hp-G1cjaKm!C0XHPg9yI}vh<TBHJCRk(<4DM{p
#f>X@+~XI_L7Mkyjr^J?E}qKB<2$1R-Od}vdh1@;jO+d;-0KKK^WKOfS%F|*A#Xh1Y7M2tY+c?3P1J<i3UTt;;3BhA<@Kv!xHzYt6FK3EqIAl#htKpe-7*?-
m%{>7-22^HDGhOt;>+=mat1IQlu`-x^1-7SS^A`sboAZs*H9<T$71Fh&&*K|xX%5s9pUpw$(o4HmZl>Lt}Gwc7u1Ku)vmC`Hb2nSRGlvj2!ktO=j3Qi)t+yx
$$gPc#{5GKmZr!kNWLOU{Q52oZyWu4sDwS?t-c|~zn6z&A3Gen$U{&{mVfIrw=Wp%s?UUNM8Sr4{9tU3Gf?R4N4H>w7t4-DD}D*MJ8GEE`k@hw7vDZQOp%Gu
C<@np$w;*Snm6R{4aMIes|^OjaNH16c79M8i!#ZrkB8pG;ULG`GX{G+^`YfiN5Jbah)-WGY?lg$(EcU!e+MJbE5Eoo=8%J($e_p*MKNIYx8rZt3Pr8c&h?j}
nc#d!akC87xIzAJibvENRU*n}OH}ocAG~srQdtkpHLn{yoU-utNt2!Va3W$i?EfA<s*DA3tKjPHSX4wSCh{l;z$*O8W`n@qc{HH&pey%-Dl?(r-92tPJVBi3
Y2}RQeJsPYW$~!L`i(DDGaN|`r!^P2BQW$O(tdX$38S~y_1Qj#LF-qpV{J<eT(xdzF}g=XGcGn-^Mg6qiEPW-v(6*4L1Q%Pb3Bsl&9v)+qj2VQ$0nmf5G)^I
IK(m#*T-ybY)_Zs+UC%vlyNYKOpYDW{}2M<f`bZ~;^mm-=x|#!3BcM)hgBA-H-2s(IHKO_kAnH6xlhMq@$Sv3$X%)=1YLV({QYt;roY&|OesmjAmz7r8%a)h
uc^^JC+q?r2eSO<Be76D6>8AEo{KEfRM~Ui6=1T_pgk#&2S{g?UEj-#Z}ob5pV=c(VNgRS9~cAjdJ>9ALFfNR)Om+f`Tu{stVm>}NLfWiW`h(@yGXM4-g_Q<
@4fekj8JBjm4uQc*-52BBq@>-lHd9L^ZEXN?sHx5`@CPz=i_l*_qpFt9oHsTCGzm;v(~Ae9C0u>{;t_zPa1xJC0ksi2a=+0deuFt@FxjUe#U8qWv#uNFbGCL
{IRU*2p{N;sB;-*d4n=4WX?J?4xeWn+<lIE;xWPYL}r;5+_-l<&o+d^ZuV)_uIo8?o48zLoa&C}<-S=D_4BYqb(4b2J_7#xcqNC$2w-0RKtpmQ0trpVNrs;_
!6tHeI9b#JOhYZM;k{XS9qarhS-}c}G4vDt+h;+zIrwYW9Wl&*8hLN}!V-UkQ?o`IeDO7(z}=PX4L>uIGOxZUWM=C5=>D<6^j5_D(=$aV8adiCaw8IfyV<un
DAOTk6qCH<;Rc(4M%U|8rVvtXY%*2!#;yWK8RdPZ7`D&WlFN#N6sH51u?PWmDl=8#*6PUkEcS-(fiF&L>lLLi`(k0={aL9wVs4MTC3pq7!li6sWxihmr#h(?
X7W|g-VhnQ*y;c=2k)ryy-LV>t45uos*icQQNsb_IG9~6BcVDUj$yYe2PdAXU_Cq7v*3pdG_RffZg*J;+EGtCg!SF<kMB&C9Z50r`Dwe=^G@K>`$I96WGOgt
;I(|UQUnZlcP~>1Uct4U>tZgs;m90#dhQOTB(lHupDsOQ11+olb=z(CA#6P7s&T0kMcm4~4*GHU^jU^2mf0H=&2JoK_Nzkiz_E+=ZUlUalCh*L4#8&cqUN5B
7_@c1q6+9zK>jG7>9JQ)AX8iM>vBs#O-E{YVwycJ?HywNBGL;*W1C9DSU)h){$6F?P=oBt;R6)0_aU%q*%DYy^cQj0nM|txy{D8#{;)MEwLLydnY@2Z>_d-#
K`(i{a#5EWr{~;jiCCT#la_b_{GYTLj*etwV3O=?v}8Iix<}`4k=tR$QoFjk*Bn2W7JqJhB=+=8mJ3{+o`^`iB3D#yhFUvwgGYuAc)umO==Sb5?kab>x||I{
ZZuUF$1njKg*Sg~ddFe#Gv~50k=ty@&Kqg3SYqS1(Bm>y2b|c?x7uQ12|B{!%Q{al{CXtv(3R2%yxlugjp|wuo%t?m#O{fO684vg)@6uYZTNOg(iQFGAqnu6
f~Cg`lVij_?L@{uS0zzgR=>~EMjj3|dmr`I#eCGAU$*`*5DjjlQRO3AIk;K;-bneI7ZU&fVJ96&l#olvT_W*7Pya)&e{n(R&tPb&j?sr|WaZ7T5q{`qrA(=7
C*a%ayPE4<Hh7bLN&KI-7C56g)q^T^uxqi%+SVWlw;UbUuaCzPefhT0emX}4Q9bUe6H7<aPSZ*DnL^xRDU2@uW{YT#N6qclPB7A`W?y<4gA?~zNK|t?5TbI_
C*q+L99ZU;j_D?&CbFuORK*b~vC>xtqqU%=vVLmJQwhu~!2yOcrkLY5xkIt+L(GSaP_aoXD0G<PYK1wJBRGwRH0<z<WZ(U%4NvF<_J~zKQpKEbc$tB<4uavm
-pB2Ns-maATWTVp%6i%(_@Xh^2Lwk_sNHa}fUC03ycpROpFMgnghSHdheNBY1xUF@PKEsPL1^)|+1ExBtg2-=t_$SjxwE7C))zOF=HEK!sUL#BKT7F8Nru8K
`}|LZjVmzeH7HjIu?K6x^iwY(Cw!8s{%Dr%0YkgAaGnhnQ0K;c@0pB6KxU($rNd=N`W!G)sS86<<`?cu9HAJ$*>ymI!W=V6;T-;_OdzUMDxy&63oFjq01YWO
u>FZ!qj_Y4-L(w0-+5KBBO%)HoV5%(8f+#<B#UwHQ>E|Ryb&yZ3oTk!+(J(Gceb^+HE>PNec+p2j*9d_YfO4##NqDL#3wgwZCKJM5$BnQ`;7Qxof=|WW7qf_
%Ah<WG%gdX2{l3P6g`>yxc%IwLB=x|_5OmH5mB!2ekdL8!RP_stt98uF0L3n!X@{X;10PFOTyom>F`Q9d7ho<@26`HP}F4CL$p&@m9XIQ-<%qVU8gw9lY<b^
J9jK@6~ONId4r?nmLTmVPm&P~!_Z;n$MoNAFtV~M%lT?1ZdQ7*3&`bx_u>nOJxpczef{SJN1tL$xIViq*%5(%*Tx=ZdiBF~Lhn!CTXmSvp7rWBaDsvaUx36u
Hxzvi)f)_QM0!pI%e0mYe0gMqS`&&9x@x%?(ozHCcP~aMnWeE$ikhZ9+Y5flJxQdDW^lLaG@CjWirZh_{XT4`1mF7a!KvhqV0&RNurEOg$5Zz83SA08_P-|-
^K``DEw69dJ<Av;*|hiGPBg*K5;@fe_3k*7D;>}HNfFHZSVO7xEl^eZs)%_|2aFzfwg>vGaA=Q}&&(Zls4J1H1u?1O{NXBD=_wtYV(X|GXI8+6MbqbkHhxf!
t2)kCVgzq@N|}P+)`;0gz<I71=tgJ!o+0M?jaVLqeMz!#rr2Y@tJ4%m_fNS_{WQd$U~X|p+F+qLv3>8s=P)gQCNJ_=2UQbt^z|0j7+QVE=YG`zb@OL66UlW^
$J*Itc*7c-0U>%Ru6`JGJe+B*V}Wy3Hm++c=BU{FY_ie76W6CG<=efqv72jde`A6alJ~US?q)Ltg}f`zB(p6-lYP|(vxQJ>B`fR4t&S6{?mp{pFG6K4wL94)
4C>Lz%udIharx;;(hE5YFo)U+9r4!3li7Dk#-#*s*GUVP{?>yv9nV71V=r7h>l}Pr#ufVC_s|-}+hJ$@+rF~DM##MNC!S*23cqqQUu(pvA$qqs+oc9QB-!Ph
D0@u6hc9W&uS)gsx!uXqz{ebuV)v{q^NK((y=kanY6!1?A`jl3HG<_T)BL?SJvg44w>)Q;2LrBX(n!B>n2ko9PhRsuJKLKZ@8qmeD>=&fcSjB!bPCDFPpnaI
s$@jfmk5`e`wlfG*kb3wMQ4dsMOe4}h;61MVE^ph5B&pLII{3a?<Bt^LO8#+dtNg@o9jTjU#lfz)=pk|w%-J~eI_3$E)dscW+39sVhP9m(=G&S7s#x1wH>g1
hSRj4gky&6pq5-H93yE2MtVJlc9rY!T3rzhi_-?}Jj3S3j3KnuRzmNKW`Xm^+KZtxZpiwVA?zY95BuZ)aAV#PW+gMsRN1OHv#U<A@S-?UM#&oL58Ggy&)v3U
(-#OAn5L;U#Klu#{7VU{kcm#!l4iC5b>%Q?H=z{%o6bGs|BOH=+t+7Il7m`M|0sXpj6xc3u}=_(Z3Q}*9-DzgI4w%HLje?w@&)0d(%^IHVD<?#heaX%<=@2o
*`{8SWbt-Eb@ZOZBhS2X`E$lTG7Saz7Mjpzm}@||Q~Xozd1F|=c=+}xzY$FA)RmJq^&vIWF=trhiWsHfGA+MY%x+9Pn%!<drHgoQ=K~W=OEt>phx*{yu$aB#
5GRhwG%Du_=|Hp8qe^wn4C!Xk#lFK*n5Px`*0y1d-ql0tCe9KNkEwiQy4xRLl9<l?h^R!^7tPLKEeUu&)VtEF<^nU5qsxDDO|Va!|ItvBE8OI{B=3ltVIgJV
8-=(YyiQ)=7@mp)e@b7**t{XA4-Y@T9Ib;U+e)_^6esX{=uRZ>iZy()|K;8>(MF*^1R5q>K<4I=N)=~;;mS9cqHV9@KJ7ZKP<|kcyXO6W+L$5xnQ`1osXW9e
Tj@`xyWoWA9NY4k0cg)0ANoQ15gY6OzSi6#a(c(DLtb-saM94Oq4;Wr$#<USt?&f(<t%njK_}=NwJa3IxS*b6v7n|V49AjhTsb+f0owLF<J*4D*t-*ck1W(5
!D|7V<S)!{l8LZZbzdJtQTF?MEB|-yj=5agx7rL33>6LJ6b;DWW$=1N()wkvPW<`z)Sm#E^$+X%E4tY7R#y9WM+1xYk3_5MoG_RFn#<Hf3$4tPMGB1G$T&8>
PiDy-7cYi;bN#V}<+C$u4`w~^$J0)1pMWRc=66_8lKbM}=VfUMAv<`VT5FPEu*2i3hf8|*JrF8mTse|sj^D@KGcJ+aVr{kN;p{hiOm?pR?1+s(9CiI6d08{K
33q-x=xqZji7PGj#|*%5kJ7%h+!L?j-YuTAj04s1M&#Il9Z25ZjPU%Y2G4EQYn;Ox_$yc0pUowM?Jr+_9fBjkydlVu%wmnJyiKNzaS<TXR5PRyc7VC`ja(aI
Uebt0>v+`XLOs0VeT1$ZaKGo%riTOC?tNVAthk23v{*9Pes5$_Cf`dJNkm06`-A;0>7ep<f8wZM2FA%22|;&P^w&`osb4BYVY8sXWL6=XYTE)k$3pRUd1wBi
l_gR`Dee)1hT)f@u+6p03MZ~>`8p4Vpw{Ok{cTq-MF03ARJz*(FU=O>GLwYBnc%WGPRxPrUu;QID|UF*vg^Yk9!;>DdwH90ox}5=$t1eOb#=NOsMB=tL4$M0
iO5P-ly?TOnG$_N(TwNrUQruFct6|Ut?vkxM=_c$?A8dC8{vp~5C$WStYJnoZ|u#YF;0BohK~wr#qV|;(4vs>rR0S*B32Dd$0fWm7sVvr+@p=9I{L2kNup29
lAq!eB>Lg^gnQBb8pzOFUXihMMg0t1U#h3Wo>V<1W5WuJ46W;mQEo7NHxR?$b_wRUi@WxC`yxoR>aG4hAB?AbH+*SK^m?n&b6ug52!6oq<RxecjWd7NU#z-e
RyXec%MM$RJ9xJ3+iMEb0TUj-I49UL-{<cvb;a{b9ft8lK3KDPM_o}Jh&?iHPfPQ?;JwSx)%RWu&YZ73eNxvKdPxOMre|FcFsb{?Il~r(T;(=K$9y4SmVUh`
-5ar%>rz!*#Yl|fNlg(pgcy@<s5F}e-tXI9D=*N1Q1j2_=p+~P?q)fmal#)Dmm0m;h`cZ!@>teY%nDy!PhLqLbVCAV>d&?47-*(F>3fsmO7x6k(>|Rb?Bwd6
5ODN^2XE>x9rke8i`R&(OZq`}zufN~BJGa0N~A99IsNbXEP6tS6uS*nl}TnNg2V9p=}r1}G9Ngm+9}yq$AV5be}4D48Ek&n=v{ap1fvLswVV}0P;!sOQr!yy
O~Asu^12W7b*SRRXA+^nNR_v<+KLA<9d~z<G||o_eezDU9!|Eh88+>YM*VN|hY1aCsIKJRr<oav>kd!ly}cbUGsvW0x8saF4Wr#RRK*}%hB%w_Fnsr2$e_RA
iTYP(&0npjV*L`oB*hbFeEBT)=!k(ioFufIZl!4v`RB!egM$$?nuo)V%3DHajMT5h#SIFj2`<mZy%2CUfm66G1be2OS$J=|L7StuQdTn(WJCEZFUfRqKdkF=
lBXG(#QiL2+zcRdBO$0e&jivJ8=q;|xuQ8Y%7i|J$U&UHeE%4^pq+c@p5eJ-Tv&NNty$=R+9#`zPMq*Y=`SbhBjt7w)5xp0KA?+~aUYcoH#0m8)hfuGF@=NN
?EU$m0L-un(B7f7f@D`mV!@O@sE)0?5M8pu0E24+qo^m`|58z%U-ZKXxpkTOXKt`ymS*1Ja)4eGoSPSY;9pP}^!a2OWaB2uwB$0tnzJ(Q``H7$EQRG}P7<i9
n6f+K<bmA_Pg49x96@U=yR-X&1p4Q%Y5rj{MdA4ekC%FlVE19wMo`NEr}m`Ir`>UYBPI9b&n7brOxHP+G*}>WE@h)PEd>1GtxpdUd&z3!-jA_pM>O_zmmHCF
gT2H~vDR}paGe_QajGZ4OP$9fpwk@3jCfXpUg@JYOSn6R=xt%fk(3GN-QjF=(ZaaN3Qg<%dU;1}arF5vo)Il`R6Uucn&6GWx7Q8fH@~If!(O-YrkzsEh<7)=
8IMANQqA3z(n$Pftl-foFvKk9*OmK8w)jN;?fYdSkN<kveZVfu9Nc<gneS&j5tJhAx|$b^Y+s{*XlGrtT%D9o39&)!)dTxU$6P@tN%O3(Hxwh!C(rEmQ-bB`
1LuU7z3}fNpWPUT5vns=GVSy1u(~SCdoe5xQZ5-Xv2$j)Z4;LFbIk%1c?LEGKizO(oB7B_vlDJkT-d$!K?7!W>@^;Bj*v?irKak%!@bj6h6gyD(Q!dC>DJF6
>@%m<E6Tru%EI<JaRztn)&KiqkjM)eKZ~l4gkA)>gf7i3K`Urq(&sVW9S*4~d+|D0qacOzd0kXA(GR%WC!Y}TTa3zf`B(_X&(NE4%Bmwf{;yM=mJRMR8T2*w
nc=&fb7V@nG?dnDRK?qkK%vx3S4^!4^Ad47wRI^B_XPCo4Ts`2<G#t9&wltCv-y>r#}}p-Ck0g*^%3vjWFE~Ti+v9@o@rkQ#9y@u@wbsi@S_!pYu%NJoxi)@
NH6NcJdF0^2$vszyd}5iqOb%t!>V!?F$c-7<;^7=HV0p{6v^OIS7<pt|Eq4S0Hq_pI(nU)P{T*@YNXT#TlVpk(LLh0upy~R_RkK!hVMgv+|a;ChDh&WEe&id
ow0hf8b(Ym;YQ_JO>k|xuGW-lW3{-yWQo@s&gqNi&xkp}@-KZB&8RcxYbx)iQR?Az<#^cJZBs0LI$p&pt%;9M3O0YyIU_<NrrS&DEGl@7W#s()P+$9`p_;)5
K?B>)GMn~jrTRR~*c*UJ%qqP6sRy0P15Y(ct?;Cv?tX5w4Fs!C8q3^OhJ^HBz%LVHFkE&!w!iB-w#H>;Zj|YPJb(A(7-<kbJgyyc-Ejg_tJN5Xh943FUXeFs
Ye6X2)>qro9cg#=eqTNijB$a!;vgb7`sCJ{DLC08cgsP1;EX<E-0dq$RYI`+=*^Ws4R(09AopWMEETGvQ(SjT{>NFNodR$8l7nFPJy~eGJO$2Q$6wG^dEs)J
Nn=HX6R59$YYr{7#^VOJry}!7_-r~B88=~xtY>kC3m1r9CZ5lDv@adocN6qh4P8L3#=(5A%^%YP%sg*yXW&fneT!#$cCd1fJo|N=fVH$2e_vT<z_p-fxO&qO
?%tnTCl}J-d~SC=(`yfW+Zh>S3p2;A*P7PN+m)c1IZ|x)D+yH|P3#^4VaRl?@7Ege#>4?Z+o)4n*cDFNKam%TeJ0Nbho-XO&de&aE5{2h)_-oBM46+ut?5We
rwXJe`iI{UIbXOx$oy5!4W#`&H2%svhPV#t>R%n9xbt3Vg@f2r1@FFw&W;x%zs+5e@~R`UO8zOE42{FHnzLm#B@~r0%H8(Gni%D_f0gT5gb4G6j?=*zc+Y&k
J4QMkA~aWTEbcc)&GCSZwqBw)t&~+w`A4F;bb)<DCKtc;O*7cHZs4Hi;eUS!MrdX(=3|?3!N$hs>e8qSe79|>xRT=_Fp&NG;i(v+7s#e)8fRneb^6VsR$0th
rk!gsbp*?6p*zdG?s$1`^1LIl_kJ9{9!SlkggEQga_&2^u<6h#rj#hfQwx>{UwX_@%}yimW;zCsoj00nYCO==X;(b*r4T{{FCX6nPPooRs(7TO3*&o&XlHjq
a7AdeWHliMB$vK@SR^Ij`|Z8|xK7^!wJ1Z^qe6EKei?hk+a`}!QE`GU`q{A1jGwcf2*m?V0!>$u4RlX*rC!?!K(e8Rf8Xsy6sd-G@a-i^TXT_GYjYH61hQB3
Wew0zvhP@{aS{wFt8}JAx^ZChu#1}b4UFEJf4TBZ1rGu<50sIIB2d1p?EJ7BB$-n}dx-P@c%+!8emw;zBah{XU5kcplq8p+R5Wa-=Zaiu(jXHsuoLar2=gl;
j9dcE*vERrB&s7A!$~RwGa?~aCO@7g6k&&jJp?<W4>36T$8QgfM*xm3dWpv?CcvUxvseKExN+{{A>~XLxR3oH{T`J9r3I#F?Qd4#;5Q~YOdg9aR_0T=CvT$W
*NZb}ldMo>JJs59g2>lYQ$E?}++Z6h@_FoDG1xqV;&PmVu`T*le%0g_S`!)j72buy{_Bk{$F~tEGIu#Rb%lVCD?h?RXG@_~tL&v$QT*SWK6r(}iL|?rNcmpc
T@wr$E5XbPx@gSY{(VuXv<j5VM<otNhvKK$wHO<oBFyKPDHKeHK;&4lhrLB0mQ8uClgg)|@~VAv@Q+-qOKT2rZd*VgUCgg~*#uS4zVf`@5(g8jD>lk9;Z*xt
*^RXre1a#MmG*ffVZbw>{HHs9jqgvq$L9p2{!`wEMy_HIjv_B7-QoW_hc$>j5RAI-+iQv3e2=L#NbI*K5+n+S64-4aEOIfpzw9bnm-@;!leIzF^T4a(f-ZQJ
%KnOy7vq8W<D>>7SM0Jsz0B$ygg=D5@3htNSXqdjYrIL!RjtCGbQB)2P8oA!9B_j^;eqD(NFb(-SE8SDJ7Bd-VM*V@9^(?`A%r?P%uoc_mR|xMF2-^kEcHMN
GppDuVH?CZc1;P&)M4Y6$+duAe%Pw?r#F1-frLTh%6Jz$T-`mgN<;KNZ^hN9-zP&Ls5;<T%%O`kDTdr@3!$LO;442VYJnBqSf1<$UKn;co@gl@1<#AIzYl7M
qcf6Z>`nb;REAe?pV17(r?S(eVcV`yF@4E*G{XX|Rs^m`#N7L|HhZ#k|5-%Kyyw+%F~Qvay$k%r>n8}3;^vlT5xnYJ({sxKi75^*!&=mE!jNL%+1Uh)4E`~_
`duGsRL2LCXU*|L;91Ztt1_e_+2%ziHDSg4=&0`tSy)Z2L~}j02j4@b=hN51AVm{OkiFuFD<vHd_vaJ$+bzRB!A2|O4f_WlB3HzX$SoPf7(mgxTU|CQ9H;HH
e9n+~!>Lg8L!PKP)TWtoIG%@KJ!!4`X`C;Zp6$sm479?bzXV#&F*T6=Jv{xZ%pc3IO4;r1sDnp~tz3jSC)>fAb1LdtIO|x!W3Qt{^cB1@WVOMy=TUUb?E2un
@%MKEWdsTnD}P45ItMqUU9xxD)#1|UPk)az9+iDV7Y2V@BCjlHPMh)!P8X`I7mcRl&wG#OH>-m2+<L|&o5))fGM_fT7zLmw9Q2*SS8=66d)y&Y4b8f@;)U1T
G1OW_AHp1hfe$UoT=&iJC7H*r`I{@ccN0t>%&H<TV3|dUISkLM7Vmc+4a2%};i&CTO_;Mb<~O$nBR={|4g-?~0*oA9URPJc9S_z^ckj8vR^NyshT0QX+$ArH
bt!?S#CcRd!yob9r1mGx{Sn9b%rS}Rr))yO@-u%dfYzRGv=rtb@67x3C&C%0LdTw2Y$##eeY%(SXaQ^u#G8)EnWCXej8j_16?Ss#1Af-oh?Um1DseD@2kq8{
l9&=O$hK25MtQ-rxpTNxN)1<nDboaqc{5kQQKzbJhihN1Y8aFFB9y=Tbxf`_`uBM6<KJ)sYxN$3Lbd|@ZJke4O!PxYk80DGGd5t``^YSZ+7u(LgUhi+=kVkZ
tKs`;Q%rT#vS<-KO8>*VaW+RGOg^2WqgS{K(H0M`*4NtDwEFdN+%64zUG$fZ<atBeuV;8;$pHM>FZ5m7EHMB4@^SKWVMunCR;!&O;0%ji6Q@rgZY*6M<#^_e
&+HD#rtPW_TuAP@!EJ<$Nj>)?Yq_`^UM)eTB!k1gtCantp|JWQ;}fdw3p`TWH6v&TUD?X}y)MbnA?G>xNXZdq13jOP_YwPQdY5ypxFt>r&1u+=sNwP8tkjk+
V6aXz+?ZgAenWYc%OsA_m7mk}KGKCt-)3f8m^3i6*!S?9X*!O1{h(Fdo`U&zz1j<JED*(75|C8wir(kkhg5t6V855c;+?e@j=I;l)wvi#jVtL&KzJS=)V7H#
{jo#1Uy`vEcLZV>RR%<yME}e0Bbg)LRyZ>-USb*dCp-gRKb``Ap(7?-WbJM(8DZCYtF?ZxHpnVEj^ul|Vl>ie=fsf^s5<`m*Fochj-B;gDLzh6G>GdhGbsnX
&Bq1F-)VTd6&Pxql?diAz5+8z0;ZPa$%}gZkZV)Oq*)(<%-xlV%|fA2mv}2gSPH?79E~+k@<=GGwXE|+n&PE9?W>uM6m<XjotILQh_@f_gxaKsU?NLTLpL-W
u36=cPcDQ*Q#)nabtnYpt7i@#+DU-0f$tXzCn9IWOgrDa7Ydar|E)73wg}z*N<r?a1(a)b$w>Nc;{(TL+K<sVc$IhhxI_hD$kh|`{pL8i<*=C)(G1sd!<TB_
{<y<(o6L$l6-IJCltX*`Ku;F?Y=^l3Dk;sh`F8PGx4HXDQ#%fInnvEj>yc=YdFe33?~W=G_bYCR$~d%LZt^801+Vyh9vOxEL%!;;q?B0}tSqhCA71yyvimTj
#I!e#Z6!;OCgwwh`=3a0aTvPq)`(^u4~1QZy_`OME}WB{>(}1M;hSs3<d|<VR7%E0eAqmpaLo3-QHU*y=4Uf{z9=K6>7>gS(^Om@6yWn?4#HtqI%)<V7c|dW
-yGD>hd}87JGoOL!X~6WC#VX+^)*B5`n&_i)AIDXKQ~}|>am<|Qw-b|Wu5oCreL6<!Cn7V8fez|A198a5q-toB=3$Fyu5dN_O+-&NKs(s>B|^2WxJP4+>VEX
Tkg4^#QVlre9e!Y&l``OVyMJA3L)rs_^xz+DXu25-0$p)L8x7NTSJ37&Ri?E7RyLN8^5B4nqn*H>Mt?+>6GBqQpg!ge;q_<TkdGS>BTe!p{Qwp6k2u|U+ecx
qR*Yz=g(Xw_MN;GN5>w42f_t}nX}PYY8nq~Pfo^UaB?GI`3=fbUr@07jA3+-wY|Uv4?NSnpkK2b0pB|Nz=0Gms9Xr%|7}k^zBL`3`dAc(7WEhd#)KrqxR8kp
7!Z5?dveZoo)Bn1Sef9Fi$ZUgxk6Vl0hV9xeTX{|2i<!Hmw0@O(V@6=AmpJdK0in?TDDh1hT@yM@-K^!)NI+6J(i14d>$0RBL$$fKKSB-aTPeDKKk?h2?pVx
XT@us7%a@13n>-4;mul>kv&xy7R2Lu$ZIqIo71$9!quk|62bb!hGt$h9Y5S+Ij)2!;`!>6p}3G+aC}l1;>Q^U)3oJFXB%RX`yy;NnUXaYpS>x4t>K2hRk0oP
$HVYIUD}(1G#jRrGh;T^HW1Zlr*g6Lglf{A6((K+=BAQ3bPIDqc|j`At2Y%AdF&P$YmT_UB9<AD6#{a<aItSbo}hV7cV>&p7y^fq|D6f)fHr$&=vqf0v2R96
T)KjBzFNtg)X@jL$^*d}2_C5YHn-_nt%d1Zrw#=S7=ZU;<C$<)1IT_PxtrOVg+0x*JDEGah<c%tCUMvs?iK<^O*8!wESL1@_@X^dNYk>omIXq2)$Y2|Zf~46
yuca%-V^cRIi@P?*7)tObT-k!73NRf+U`{mb9+*zfi+GCj%53j)312o$GzC5oK9oRZWNlxd*-3wvWLb2{&Yw$O9%gy_XQch(Nhj@TNHNvD4c5}U|2*!ePSbs
m<fNpZkZb(F{aIhj=>p59laYbuG&NHn_L><f;SdROr_&gy+M6C*}aLFV`oeH9PQ}U@o-O}^X4;uJRvQcT)*fB#oM<$<P@CnUQe*$IH?!bB`qgSr+|pF6KglG
+hdVbzbEX9KehzZQ#Nr9Ix%h?Cs&*>dSA)KJ<bMFZFSpcE3$Cv!%0VqH6!>^*gV}E>xeO$O4rSYs`%)qv~D71i{iyPYhFhc3<Q#`$i(>IHNL!#sExqX`%?rB
2Pepf#GCa>Mj|}tkA#AyC(1U8$%YNDp|)93=H+)okddxd>bayMVDIbgam`@p(oW<mlv+R`>PgM(8Xu5!Fr7ck5CP@$o`i-hOIVuum6*(FqqLstgNjlRjI6yn
n=j~rzMWQYh{_!=_^CASJ`6%?q~~>k8B^?<PjY*9*A6D^KGke$cBmS?{Ip~%4CmTscBzt{#YKf}?>sRh9D6I@{W2*76b__PGY>tGQ|3g)<1Pv7Q|wdG2h#AQ
d{Ujp+!l9)>y0O_m?JKQr#|kvFTRg;t4?pLVCbsYoyCWCm?@hS-#1U(7c4A>97i(|*}K?Fd)yob5A_xfeRITc-zToGH`Ng`v%^$E>5hA`AE~=T<ME&&A-XS6
8>Ft+51!d91-sw{&B+!^$OOM0`CX`i3voX}Zd@nev63#8)p#@nC$e<{V--<v`>uzTEf&YFJ@K475rCf}V)sl|9neroE7&k$gYWq__rBECgioEEHp#Id7??b=
J9W+!S!t?w%Qf;aYIM5x`VTv72fdskXR*WHBWpfmrUg*5|F@9x+7vg%YjeeJ#6mxEHeM~*4JuFT-j=^Hz(&`h{Wgmp_?&O8*~aaP<T7s0iC}Mtv|1;;*&~f>
B9!0t0;5ng+q2HxmWh$$nXFNL1O!(bIv!H8gH@;3o2r)<AT!T>n*>>KI4F(U6Tfd~A~q^nE&!PvEWVl(QlM8gE}~;-!FW@&==T5vVCbSVOKJ+LH@>ZL282Mf
uw*Hn$d`|rkEh*JkH)=sP2y+DLvV`ny2B4I;{Eu}Czw<fihu{32kHL$LzRm$qJLTq<a|LAaqDK7lI`o-*et;lvQ~;@5hX}t<y|aEC@vNRS6lS>;`qCb75Dl?
=rD2qJ2e{)GIjf_iXF)y{Y&0?DWVu>7q-?}Us<E^VVPmv4HcwmC~N3-NC6-3@RrBg0g8=#8jt#-(<PMux_2qQj2vbVg*jY9TFh_1j>P91@~0V`L-Bm_lR9Og
2~HU@F>L<!z$E!AKAGHTbdxd}x%m-2<z<~mU9l;e6l~vDc}HQT_w4ZWO@GW@<uhEJ&cqX+$lWU!ME>_brqyUa-NrP$qSdOP81jML8_)M-szh&{|6*lWqleFt
a|E?~YxK-4-iqq<#Mdv<Z1p2PxNktZlKsXT&*$iSbXP+_Mt5KA-y0buUZd7AC3qo9JG;F9cRG4<#CV$e67gz~*S)OU38YULei*6+;Mtw8ANcmVV|S)^ZU)hp
OYbWGVX_ZJtY0`=_Y(^=d{PM7oJhejQSZOIZiT|p=jU5L|0uj^_Yfx)4+QHF6Wzfp+1TW=pCeQE!PPc(((1iwIHLIc{Eq?x8dkQ~(_cFvmXe)bsrNGa9i9dW
J~4x#;heoN>Jb<0_^6;L27*V5H{Ljg;dRNKrGL!kSY5O}ApNBqO+z)kMMU56@GxvrGa>S4@=^J(udU!mmR%k9MjfBO4Moo-2O_R%rJK$>60v>4Y8HzTxMn}3
EOXWw^~{W=A5JPkVaxjkS86CS=TAMRYRUq&$Ct2#sz8t@S!?Qj&<9seiZ)-PBYt(!M@2EZ!Z^EoSIcxFmgKJvs!;o3#gaL7Y#|B>R<1@i;|h3q`Bi_U6#+dI
JNELSp2*kN@D`u2McOl!klcq#VBnuQzg`lF@mVgnIguP_-JiXz{?8mLd+8oIohU|A!h4qd?l`p6rISz9$3ii{*L;cI4*S*5ZC05^pgA%8(k`YbOy_ysU<pkJ
yZST!G-6-cyJ>eHxPB9J8oypYShU2J`Jp=NOY(5L9<Ol8xB%aZcUNyd2*(>jWe{bSE7m0@lT*fR;JJ3+(V8j^DIR~*2RHH%D1IsC@pcXdwYC<ildhrNk|!=)
J`M-wK3dQhrD4azuxFnB4o;^&CD~bUz>(X|+(Pa{V7T+GHPXui4o~G|lFz!rQi98vOfVKM9M*yE6sgD>KYwRbu>`HD0Z-~ZBhV<I=kb881ecGD9Lp)JK<(KU
*%pon>|&F5zuXgwpTfK>76fzb8G6Wldp-o?_W4mi`TdYyV}5FkHv}eWdxVA6+~HF2il$}K4u?X&*uUQwjk3)BhNGL-IBfT=fL_8EAFozy8cP*nH9hOT&__p%
(9Wf(X<2~L<dJhjCILK;Ml|_rQqVp`_Ar_!95eF*)AGV)kg41{t*IZ6rq_};zWoYEw6@-Zo@z(T$%|^eW4VRhk7lkXIez$WPGQ)0cvdD27y9k$`ZMG4fr~rq
lVc!S35i#a#h2mY?dqfSKcC>JK}kZ`_Xyl}c9^IwFo#l~Q<ZnOJ+8EooS0_zMfw@8SQgGgNEN5qeA0KvdESrCHIcT+<<hic?=;7Tub}mEe-=*K2k*MYa2vN~
d`~C%IpAyWs6tSgAL=*XA6)qDiW-`x&_-5ej9+@=$JpqBC|&o1RkI;r5Rmz6sO1fQl3!MCj3FSjF3^ou*kj-D3)AKLD`*Rw;JzcFi`O$Pdp;-=`-hT4mmx9&
<8LLXkFL5w_;gvs-yk>mt?Cw}ge2j+@q=B=SAyUlHJbCx$q!S8RqW>5zIZR5zw}AT3xXkbd!MP>BZI)@nj~lgn&4)$uz&JcHJE7#qZ5U|i-%;py4+yQM`gr#
R~vrI=9`~8qhQ4O@0(w)FYMdi?GjXU!`i~M@GUn2PLf@wXAW^fb#>6j`Qb#^YBW-w=rKUs&dOHo{UG=dI@2dUI6&mHUxQU=3M`v;1db@Vpj9k;N>?BhK~xSo
1^LPtS?P>l9P&q*$@l|FL3@l86&IY%Bl@O9qQOUbH<*%s|G}Jf9)>?7lQjCR;cHLX>vJFuYIKkETxmo=_S~B>;jS}eN(dsNFZE%y-56eTEEz>F#*9vI7(r~C
Idv!74gTlWe%_omz?8ICO<9Z%SZ^?8A7zpQKZQz1m#-t1PRab@JDP#>R|VOkIczayvv$~p5QUNl=HpX=4j{k&yY>u;654)m<VAFQLRReX#;A2N{%Jd?kFNRQ
WZ&nuo43s1d2wm}Z?z{rlQHQV9P|PA)WYDSWm7O!Tt5CwN&~LPq#j&99)xsLH7hnFb+qx;>D;2VLE)p?jfo4fc*DH-<Px<uq|Te2)Qz)+1()Yg;1_*VAK3rq
??N;z&WDt3{uPE}NZN_d8@l-VTZGj)HVctWCMV3Tov`xb)KuGhDZKspRM?#^0y%yR)x1RBRZVmbvu@VMnXFwc{9EC;QAVoY;w6t>@rhVXQB}zLXQitNI^y7#
343I9JbZg+M*fhR;#&1*VH$!pg8oTcjU6=uoom9ybOk@Cv;W!hpdxal!|p}_4Rw4uDon~r>4&2C>VK}An!)VF(1B$lm%TA!dd!d?fp2sD%Qq=Qpma^#T>Z=y
1ZvfGiP;l9?xjkEBx^E?e^ho|q;rJEg@un5zs#}xCg;VIy(W0Vb?Y8kvJ;MWlr@)g8R1+&Il<XG882|E=RkxB)Xv4|#3$;(e9zZ!ns3w4VH_TD_<$Mg<^#+Y
{}jT(@O!04pfmEhe^tpSYGAed#@RV*SM(}AB)>FbhvpwrC+UxPAb>YHCAU`^N4|cxBGkI#3n@Q`<iU6d8R*#F?RQ3G@RhKI&t{mTTgoC2F@vg)MD_U=5gba_
5fdW%!On(qTtZ?LmiS!0T{|TJe^#}y?5!-kw^TK9*;IwT=LfdyY~h$HRPk>bvc`_n#hdpxOrcN4t8ngyCHzbk!@@Y-P^0{6x`>!pIX0gq3p;$VKP7p1>#rjo
`JNto@J<ymM!A3X@$2HE{Gxn|br>Fh3-(wGIEPJx<Kw241mwpLkb4}7z!%v=m5!ddAZPI(-!hJey4#L@tV{@0b8k0@tj3~bcV1gxiUpVgmc_VBt|G32U6`j>
5-Fjf9g6>d-ge#BIwvkC?8&`iK%JC=TLq0%W~0jZTKD$uPoY3;Ze0xCs0l{vv28K)C+0X=;?J%p;sMsy`@XS?33%GQJ5T?w6Q)#2qRH${!5}|Ez0jA2$0Col
o0e=K!k@=9tP}#O>sy{=OK1Pf?*#pj6oJi~h-xW`{Y#b!3;uz2vo1SSzC0ik+--`q%g4`r*sTi|*S)-Adpxlv>CYS`8H7IrYcm$NgQ0CqZvC;{4Q=_f5vwgy
xc`+!caGQxPmkEgKFP2_ZTz^e#A#E2Q-$`qtSd<Qg8mqD<ROQ8cjuC-3ml?9rQdnv3(ENK8SDSNF+lp3{hW$9;!pV9tmq?h<kPcC8rLK7imqmoz91CMA=dBn
vhBbS-gU*0)(we!z0x<W^09biJuu%W4o(9Gq4GNRIB8tIXO-3%$I|w0DrRfp#hpWvc30JLQEc!~h*=_%<(BV-Sww))oVVp^7=lm%#*Ny2)|l;OP@>doL?`cl
r`h#TghqJx-X3$pxp7XN^V?>yjGUuPkTgZ5LB*Wa-9Q|t8TQei$;PzkGx^~=0XRrWxM}2Qg2R&xdxb31!O(k6%-A6ct9@rFjda3sS|hbraLf`NO>cY0u3Urb
4U0=zKb@g&`}C)$M=Un(E$}(c--N+6gIinf-cXdy?5?Eo1mjr5af;we@E%u;Hf0M!;A*~pr9mWi9lXW9^}-fm>!a=ee=g}`uB4tBJ72u`wASw!90<$KakJ6}
9YoaM4|O&Pfa$9rvf-n4aBO;1^KbSk>R6_9Q~UfeMlF1k{-Y<}E45ms?MXmV!Zvk)ZX5*cZ!?X^S;KMhR&{l77z`-iTO1^dhIHQt>d8~i2yD>YCmxjwSsIHZ
2g7)5tG51V<qyG?IPVC%F;fg2{-lZeOr-H#9{oxYjB6Dx7iXN3vC_{!q`)SFptn(uyH}EM^v(%)qnFvx_J8p$?n*wkUWuhkZFu8Ghkh~jf(wK=2i6tD3vl1z
K*DQoH*gABoE$5UhZ{!&(_xYb+zdO*@qj7}r-yg3)4VQ5W7rOXlG_SL_wZ#p9E!(LCQtcsVh(f-zj#M~GYm%2;`&Qs1dK0z?2DUyh3n(AGNK{j#LrcX^(baT
+`*rh#61G(o#(WkJPJo`-`lD22mk;6#H+IgE^#<|rCWAQ#RVrcGGm8d2IDnd5Q`|AF9N=ozfgWgz~G)VQ91)p#PpZhpWsb~Ses~ndlRwOjkJ=!a7E(l!Li*E
g5hu)`W5h4%^Pn>#JKB@xnsb(X>ZP(aQM-M#|bzk{x_#BB^8QCjJ?pND;Tq;TM3OlMn1Ac9!Q+uHA8#L8#}MV54_$x2qhce)4Pd0?YG7uoAc8R0#hU^6GwtT
^Xe((Kye6Mm*(heDWbuCbBz1sr({G;Jc<iub;Y5ccjYFNu?VuR+cTk?j_7v}mn$}EAVRxeQCxsHj~R;27ehku+g4KVo~IQYr8mb1yA@%SHshlw9gUd;&j|5a
H+<OGiXc_9!na5#Izoab<TZ}Ahm1R+HkXf!@LCZ(pDp-{&zi&Wf~IF<dIaoVXqV|2yFvL#S-FLQHB`+ol+SMu%9BQodsB=t`gZ&&qfZ88q$^2po7;i*5p`K_
tS?kwF|Dvu8zUo^Z`OCg5GPdkhla11qJ8-n`}q(_H0EjwvC=5v^!HktUK=|EDak$G{HzK_v6=H@!UT*okPM&sZiQ1M7S^kF*YUkIw>aycHQ*RmvBG5rs(!v8
o`-(mO!s)^m8yq&9rZ3w#!w8*Qn!2anxR7F$nW_{TZprJa~>u!!PnC*QDGmv@K=s9_dvETbo7in1hm``Hkc49%VY}SpE`b$Hw<wrwJyg{(jB?e&$Qmhh~UoK
c$@fxreMjk7dAQV3q=~gZci&AqMxf#U3lS+{gicOg;eHvao;5FQ(++#W~K}4owcB`RIpBd+X8)Lou%~Ys@Rfyd?oX(5#pOPLTpB_p(S8DBAUbtJ#x4HJ@pF)
TRIuVOCnF03mmtxop%5;%dwoVtpx0%HOOguuYk`V9e>;1v_*FYm0|JO5G?LrlJ4|yLU#NwpFn9XQ1!gzCB12d1BGdS<Z9iqaDGEbUBw6&r7vs{4qOMF2lpOJ
VQ1(c@-V%$Ws2?b<9Wig4p5-uG^N{dgkQkCg$(CqOnO*aY#nw(#G`#49tU*6;m|mD@O=z^Y0K&-GK-@6<espXJto-e7IeQz#uPr;-y7D`El_P${AQsQpgrir
e{?Ys3>8;RqemREW3OLvu+9>P>o}@k@_V8u=}Co(nGP6uYZDlARItZF#Yw2n6wE#^ULHGdhJEh$ID6h{gKz9^ib#tU$o#DapFdH>uHp@j-I^}Am-V>4@U=e#
{Tc%wo6AC1I6Pab%^K9_Yxccy&_Ju#*~8y-6wn>{(oeC(70SOIM4n#t!N-5QC&WKr$L8?t86$Nguyh%w{wDT^-f<@zv7s<*sJJ?Ks#$=RNmgZ4#R_`0A9IEZ
tl?hi>h*1(CXx@!hxYDr!ecnTj6C-i?|j7_-vv!!P||lT^SuXR_!iXaJdB}lNw`z9W{0hIQ@)sdSCELNi>5@&gTkcniy4JE)Vz;pr>0xN-;X)k;;}hqABJV;
z0!pFRTjRnV-e6Eczjx2z#ioLfBdW52!LES*MT4v9f;0oc@A*f;;#-ZN$W9QSe&j(<_^}t6QR`i^s-^-S66J$yUc;`iyAxuq5j}u^Dv=c_kqv|*LjyIGmsa|
l3ZF2!>i?k_l{4b!l#Fe!}wYX<TpGTqWLW$Pa=N5?1&fgqz4nu%6i}+zrVr(T}y1UR`NbPWDKEy298tR>JYx1{Uoc{6_;rB*5V?0L83)?I6>wH`T5vp0(}~+
d;=(NFwer`>vL7^r*^1!SsS`br;lF|eSe#;1;EnGwl(0iF7~OX==_jUfX>Y$gHNfifXRfD4ny|P>j^8jmefPaEAr)1A)?>$@6QUlB#r*C`u2=-nQ+{%b2#~z
2@a~!FysmvBAK`H^*1(W`2Vt$Vj=dI#FZa@%2N(d{l@erK3NlCZU_FfpYevp!->CNN=;!tKxg@mDGG8|V-7#2=lL(cH?<@m&c`I<?1R%q$Kwg$u<}pm98g8i
-uIbBdD_UQqwad_avd~d|J+U_JEA)C!~NZ|UNCQKoDBMGj|7dTT^h3~`1Rz+9wxz?xY;9gd($Hlxe8UwW6WWQd$*haL7Xo#Y(>7`B<{zV#*bAJ#YM>c$dH=G
9*r`th9=7VC>ZDpWsc5=fU-R)w|y)HT!P)aob(=8T+bnkt!xIx{k)%y#GY|`&$ekWlZ5@WOSfcKf-!$$*OU67P*6o?RFk#mf+t~~gqkTFZxrZv&3~~)8Hd!_
4t;HKIv=pDxEX+$9Q%s7+aAbvuU!pRk3nwU_3j|uD7=lFJx$wAz}lJnJWN+3@mtyRn74;Jlrr{u=~npQm4nb6L$w3We15mFqOOg|yQ6+)>K=Hcyvij$<cA}J
2E!C~naJv18aZrJjxv(YuL7<nu-m-Yd{Hk3^I^d}7mNvTxt4JIZF@Lo{DX{!m6c&(^!=mKp$LSQ(Zqaz<_l)FPJ8C(-uP53U7e$qjz6ak=#BW~LvKOshsaww
q<F=)cNWLsP#^briLxEarj32yG&|$r>HBw$FDYTSi)-uOY2tlnQ~C8}^9FAF7X1}dHbJs~n*qgzr&zmOC^WFs0RC*Umfw%VKwms>_lE5@Vh*O7Il9Kc^vhk-
3`rlbNc`DWnhb;fJB4}W<Ij<<=NM>xF&)cSq$}x%9nm$g_q*X6eb8}}2vth?!(~d4G(IW~)M~qJKPdZxA;>KK$6_G%k=|+eQIvoSvSCMmQTjmdoQSyb;Vua7
-cL6o?1~xPBId}enF#AT!N>eM3oEDV&7Zvwh2xyI8%MMbysHOBUg?E^K`yzuOvVbegAm9&;|6{?_xHn15h&>4GYQU$z@39JqNVn!`1Wh)&67*<cy@xrvh$1=
#7Cw1=EcHLuqDf(JQ;%d`s2g-)4pIFQf-Jybpcn!y}*St5jZs-wQ-`u4{5A<>FGiK2wt9Jlt}Tz=WovDHOg^lIW<5pR_cr6hSdCTbFI*;q3xJ*)&ruCG|pc-
kdA*O>&9#+eL+|LQ;C5l0Kq4|v(!raz&mY|Vd-Tkx?k;FJij*wcO5>^yyFYTpQ$|uIe3Y@e?y)ovM3STM+oDt49-}{`%qz`LY&8~@N?<)?*Gkc<u#?#|L8){
cE9bOuU;h4Llj~+R044OuBdN`N**LW(`tV0bHPuO^FiZJvSF@sX07~y6>`-1zi5gCq3#ir1QnAHNQ*yKDA#AeA&L3pqilDmHt_bh&Y2-RG>TR7U>L{;6(5<A
2E+XCsE;3S7~KA}?moZf0J`TZJ&r_w)c48Gqjv~|cV5-O!(@igy5Rb?dddw~+w7{pO2wjmJ48HIAp@5WXB<8)M&viGQu-rHo-h<~IRA&<72~~RPd>9-U`CcJ
FfP>_4ZL%)t<Pe>JGfw2C|QI?H-SsGlQ;0PQ(*i4Y6LEg9l1cO76Gcoh|*W;u6P=8hn(_SELb0()6t9c$6~s5$(LIMjP^8?QpEdXOa2tAJf|m2Z_U&k$Z|r$
Ld{L-l@NTFaI1_kbq99~_qj`D$*`Dy%+DW|fXRlX+LO&*xYv2HJ4X3F8hmPfxM@ScqM-NWjzKuqb~g&us`+B>?#{T4SpZnQG7c&@5>V>G^HWYF1S>pmo?Bea
z^G9718ciDc-${Lcf`OO7uHW_tlOp${bMO_Su7KrSK|L>(tBb0VB+WeKOUe;2xCtq=n(tV>CR?l9KISK;Mpck!;8LKHosjJQRz6J`M19r@n4!4xbC{6c;WCm
MPCrgSYos1o_peBNNo9u&@eE)5aaHWazXB$8)Ugu9ys*Y@bg{~Z<xuh(%pXQ3-(L9jz1>u@1bxKj!&oSaXYE)iycEMu8_Sa6|ExnMiZBk_DM73)0YfBbMl3Y
*Djjvx-euUd~jE8Pe;Q)jz5QOyfJM$t+w}tC%n_ugf@(CV!(ou{pq?V6z1sD8T5Q{wDyDNbN39)x^P8r{SAbY{;OXv|BtBijtA=f-@lB?$VgUXZ`mXJviIJ5
@4dIojK~VvqN1Uc5u!qh(jX*}QmL#aWyR0ukKg<I_j#Q2IQM;C*Y$c{k8{p_n}*`$7LVw=y&Sib{_RqeUm9HQ1yw>Z3m%(Vyk8$i!B{$gHTPaIyyO(>?xlKy
y5!4%r@~1n_ctXE@(hATlOwkOT`O3`dTG2oH8EP2pMGawH7Evk9d-|%g!$;<12W%pVe~vG$A&2x9>v8yQ@=t`m3>%gmChfVV%MiU?`J@OgoWi@-(KCbOkJdU
^$?e>HaB#fOTc9iA7yVJf-(w<vlrY7k(oYr<j8O;o;3gA{qiafX?})1jDF3CiEc7%-}ALOB_dT43P%`Zs%YBfrh@jCUu}|RIc6Ay?2AW|VBJ+T<<DG;)7N-@
xpNjEl%90t=*_)+qsK8%z?cS|*}*q)lP9r5e}I;_Jp{Lc>n8UtxMEF}=htPr25>ird8fpiAliSRIP7`=V&*ok(C{Qf>Kc{bN9`n(o9bQf2#QAfZ(k2a@>F=L
*0qnNxx>1s=UZHC4P=(b9Hrf_;D}0rFlk9S$b$waj&>#>(S#DZ(YMk0LG#C&T0WeDI@4HAoq|nsmPp9wGq^D4KG%0G5n&UwbiPhz$fdi@$kv>Ip__+AHr}`3
LrmGDKh>V-QLx&|a*xD-k&<dfhYxP<<L6m_oQwq<Ysz=y8L(|jiV4b%LsE*U!{!N3uw0Dc)!EI*Gs3$$tTY8kQIEQL)1v|&4``PjpU*(*0>PTRp9@wlKWe;h
kbsq|OXCMJvtdn4)s^``3H$&35;-nk4WSOd9W#;~NDe=A$RY5DFxzAfy=@^{-=)1GtBXNoFSEnnixG(4ea?9xGz&vpkLlKhBG5RtJ`l&80v_UiZF1f;nAP8r
tQqh?1E<%;4#5mezP9jvQl1Q|LkbrK!_Q+wm|86CiW7c^E>xMTl%hoLEtT!b9K4N|rAo}G+RK5oSp{!SqTn*UQtHhp{Ep*1=x7@O?lRvM_TVT)b2B|}AS;1@
Le0}$!qETx`?<d?{hg9b$R%0p8wUhqAu@!tQL_k4_usgTS4P0eAm`S<m`F_7*9&xC&4srh@0z2)NpyLI2e&^AhPMloSlg68o>?1IwhTt0hhfmGs8;~?!5(j;
%^h%LdT3IM#2iwU)0Ha$NAdXfzQX7GiE!U~NVM}O2fpd~-oDMwj^u^ow*2W55V+U$rP0X<ZWiOdLeJQ6yL`)q=foj6|Fql|KOzIY#GhB^^Ca+g=h#d4JPHiV
vpfh%J%F<b1Lt*Dg^}St>pc99AI`N7&ytV+Q>Op7a`fRbPI%0AJ!xPYSI&1QO53U-NAK0QiqdVz(KNN0oyWk4!0UIS6jP;fx|?2@&7T1eiSvSWBiOK-bCZC9
n-KX*GhV(?9C-btkU79=MLDsc`G>jWA(+ZJX>jy&;lkF3)HZE8MA1`3H2*k^q^vuZFNyvrSG;EWVcf+8kJ97O!JnuRvM-C9UGM;!xT{|(@rZzObCHDQnG9Hu
49OT2tSEn>S(jrMVL{?Ph3jI5^k@}t_<P7o14LAz#^2h9mFGr6i{mx~VD7u76Z`2Xwkc~mtlOn9^2Efy@!5OjU<Gl;8)B3o5McBgZX^eP4t25W8)68)X-Kvs
KZ-c5JGa9Z7*H_hUc=AIjjZovW1J)muwTD8-xMYa=QK6xhptTEe^z%WCs7gu+Qu<L()%#b^225Kh#1b@`%W4>%mP8-^nD*|xiR9aBr@4Rvp1jSjY?G{U|dZe
xZ}@-`_=8gEdr=<f@@^5E93|+dHu7!PC$-BZ&#;V%s3HQ{o=2v*iU73)4K->4=_Wqy1dQy$*yv+Bn6*L4>KYaT~3q7zgJ#sf01ODLWG$N*;6k#kHL}DUq!l!
9`}B{cR&1lLs^WNcscXjnzEoLFT>0Z3u@x2F3EEo#d}M=w(f*Q<#WA1wyYy5_k8M~W~(tRzW>&)v&`Z|+uajRzlZhk&f1<hq;?-NuX*U^x_TokQ#2qyo)3ST
;&Q!P#c*|CI!VQETe&2fxs92X3Qeo|wl5M0(GfV6-#f*Et^uLMtTs}RWj;D|YI{XFss7ssVIeY{T#D^V(p113%I_D&Sn2nk!-3_uHNs$cBzKgDKp4?(-6~pY
_Sl&JWm?xPgn@gf3W*4H&{|VBcY{F<ImEFtUrB}j)9I3e&gk2#8Yo)Ie*B_H1?f-kHm83jLQ94Hlbh!bqED8ue0$~)7AqyriDvD?$vgiX-&Y)m<z~c*8yy0e
{TS4q+$e?R{;)z@T@ldl)3*Cizy_vhALU8$9pyImdX_l{GDNq_xV26SB6R9_@@Hd4=rIveQnHwVU9Pc$=b9*Fbm}5w{p-<1sqvCXX+t^m6EtY2g>m{p+|4t-
W_V;)^6bwUNoee}9H8J+MUO|X!@s8EfYkd)-FZVCk@V;F(`3env+>t8#)7b_vif-XGZz+~1&}Rw8=`bn{jczlILJdAvrP}tqw9+U$J+vN?0?k0c5JUM9OmBU
1Scus3|C8omb3u;?F%fsi9|s(Mb%|aY>Jzke{<6&L}BJo`>b`-22VH!h3j&a@mi8rZzfnAx>xh-6lewDV{vEpXf_Y@BF^8~Z$bouPgODbUK9}JjxJF-%YxP?
XI`CZl*Wt@tEZTwE?WAW&r&St;6~b~DUEb?JbSnALKpo}T=L%4&n$^V{*0^f>{W9x-|=n!Lf`_yhqv|Wiqzq3e(7#Mk0k8r&aLpVN<y14h@$bR2wr74g>OGv
QC?1u4^^3#Mq|h)ClX2xB##hBJ)c=rZq_qu_23dkPXJ?nQ!5D`P0jGD&`^WQ=5OiBq0<QaJy*VyqKRMhO8u{*WWgUc__K}N0$S&vTvV!;!_24lFE7tZV&&Qh
^EEdsJW?%tq_s~JOeP%d_8|uN#m`a3`i&m3ZLD21-XzEiUf>g2VMXYR<8OO!NW$tv#re=CO*F9IyV9jBgOksF`nnj{(X(99t9G6n(V|vA6An5<RouSnW0?*F
RW`03an^u}xZNiV>B0Vx_9mB;Bp9V{xIdc|$2+><cLYT&s1}}UW2h8DNc3H0XxZb*MUPYBgx1J-Dk$?rP7I%{A3yQemcS;<A6^M@IUMvp#>}%U3^kADw%5PK
p(vTrv%x8fpI*POO_FJ&jypBARz?MxR<G(jYNWv6{Vb$PLK|Y5CtLNZ<zek=IOAO?i5OS+@~sMG^l>>6cC1rkr}B-WT96|2?kbRU8d&1>+`+}Qi9qy{Up~*T
rwaa)4uZ)yX(2J1#Jn^ofB-=j+nT?Oc>5$Lgq!}ivY|@z#f{IpxKexX3a2d>F8*oV?V8a={rk_cH&&G4EBfz@&9W3i3t3OsEQI2?(2dyb0#O+BQ~i)mlYppB
$fbN8O3+!qI>tax4M+KEMxtl*81%VnKb$q8ymUS)BwK!2nI)Ji|2-!eWF%V`h5NNJKjK8_m&p#zt89I*`kCQu#q(UslK^dojiyKB#SlTKxBm7H7Z}SI-fssg
AY4nLD{s&hiG7J#&r%M7km2Lf^e#2JwiXTD9QOLMuv%bgo;vt6a{9G@TA@`VeMyH=8d@BTBIF<0af3QerN2ua9{yh_XaD?AR#TbaZks;>CF5*A!Ln6l<<{=9
cS~QDKl7MsUF(%Y%PYf|`xyTyv)%kyGF?HpmtzGt?6#!gB;GY}Ex`ddB^?}Y?yM@G{pfURA&(r~5?ZYH)6{YOh0F5EXVS0_Ag8Q+APDUSodJT*X=U-5SIOM`
4tTqsKsTAE22GbgrMkq7(4kYOSyYfmH<|fDLicy&bc)K4NlrAl$|ZClrcVis1YcSS%o%Vm|IN<zL)xfVI39eB>o|I)%PI8EP(l6M%h{1MMm#hY;Z<?u!*Gq&
_`&n9Ab7Rml}h`LGGpH2{#}nF*u7?%*Oa9Kqi}tikSab1*@+R=*8Njv=HDEQzsv=KlI-X7E<cr%P6$woU)6+PQqh&!q{H}C+4MH>*!+L{NR!t3*Jt{3L%aTs
3&&x4{F&L96>MUMtLB((hv`XVy2f2T7%Ps&9gU6AQgI+<PEP!lIvgf8<X&4Hh8UsYyHJDU@V@l4zU#*?<yX94RwZY3;6B_WRZbj^0Bg;EP4d||=VdU)@mLKv
qLRvvs!2lnN}>)Ct3HU*6n+(ys^j^AQ&Gl;?I6gke~o}F45!-@?+P9<!8!Kx-y7FNpm@xW@bsb;$`AR>i~G5P?DyBFw@ymKfizmBo?jL@(y1dgPgJ2){f|dG
!4AnBAB%1!v!SF_vXE#?6D<=OJ-H&>*qmsrH7rp=@ms2U(+no?5WI7a^Sdfiv)%LlToFOHgxs}XOLBNDazbl4K@GjdroS|^#h_3d=*QS23x;(e#z9?LL<?qc
Tcw+TZbfyY>5LY3Qr{diDO18A8-d<i8g)3bIq>!>AH}WR3Js5&8qnKj%;{~D2j7P@wX#iFoRF4lxcBr3+%pq={fku5T|soeDx3wqzI_ydsTznnz?Iy}qJd*)
VyXHn%<$MefIBK_N7+?);$d>84GM^*O8*^@!|!_U4@xB_U@*>H87iPe-RPgyomgOX{jK>@s0QK)+79To^CDvX<f$}5Q~WZ^5c3KH?r6!X-QKT<+#jPa<<2N#
0D2eKnsuQ=-#)bwq6WKmJO3Jcb1aLh^(=oD#a;f!;`Aq)a8LcLbXJH7&4eBI2hU4jRn%3Av{nOKpKoU5d0QdDadT!VPz{;MZJY}hg7EvCx3TkH7U9pMszN`?
;r89HZv-aQVK(|cA*+=GOtkkr)9g&3b6kUYUy35ag!#YbJXJ%V&3Dl|2k2mRNbvZba4~G|3$17VrUqqj+Oztopq#Qn-d{)tlM(lYe+P-8QRQ$a>po@Fmkvk8
5vk)zhS?Nlq8Ki{tUt4>qXN3Jnd1_lwNT5H_UOxu9_Bt&8(wZXff@b{W@#lcoFP2ga%s<J0^(W>7;DwxrOCb{V4)7S-oaq~auF=1kT&>P$-ttiUcYsS3zVHl
HO`K5q5Fsl;nNxokk4#6t9@sKI^&4Vm$QsGx6e_r_^KqVo8%c-{;9!*>9GQ-!`^eepCkTTPaTPix(kgCrAYXr+I)6TR?2)L)7IW9SS;{i&vR3Rev)>v`yQwK
f1Q4sQ&GAut&HTatP455>fqL|I)3<z9R7wC@qA_1MJ~(5UXmpd+`k$t&|Iqm#t-7*>T~Q^@AjfNzpjYCi)>v1lPZu9(^`3|X^x8#BvhuuQn;+#X)902i758R
EpN^r2L-p;r=KRekh?n)Y?P~wuTjt5sK@t!K*p#^_p}M7$bRx0k;x$_OuZoW4mZ+HwTykEb3=VK=k$;HAiOD7G`O_nf=<<a7Yew1@shl;V`kO@${gd4Co;V-
{`565SC2OGVqg5VCpZNURkAj&eO7qL8b<fj#tU09^=ZpjUEpZz@cX=#8v<%__iOvwz+k$&`+J8Tx*hx?i<g3NEi8vH;Ef(S!e-v|R+~fPZ!5nwqc7eyosIpY
rH`KppUV&RJD`Rqw}|VVE0()I(rcNzz;rT*;?-CNWW-()Rg)UxlS9Zr;zc#2cwA{KK57B!x&5Ecw3)-_{=A@{us42vWYTwI_5{K3spP;_2jo1_J?`hIj_yjr
7=mF#<cd6f#vYXk#l!DT3Ho^B5MvVgwE#cFetad<>=26c!uwA%A92O-O}2=i9%*<yx76-*%M=c;cWmmqgb}-Xqw1@r4^r$7YXz0L<5pCz^t+clShlp)?;^Fs
;UMeIS9^V%@scRZ?1(B_F3&v}P|3mPqfrm-f;~XLH9zOEYk}tMLj{A49&pO;%Bj8>j_|LhbvJy1Q0inou||@Dao_oP-UWU1>801Kv4$gf|LvLmwc6-nC%+&k
DhoP|o2qaB{_p)x*ACBm6uu4k6RrPmzh!B~-3V^CJr13nVHD9uhn!I8PZMW!SOhO|R9u1U*!El}X9NUgQY5B%++jN{(7(h{2bMLlt7bH&AiS$EPkms|7rrIE
9XqWHo>z+Vn&l2)opQQ(_hkyc9v%41#N~qH6({=0z8OKZ%v3k;g9C0u$+X7A0SO&)9ZgSc5Yuh9cz9O_Nwl<&c^KSr&g_Ca)3i4N$dAlE_in-Q#RGM-zZ`M4
uxLW@mm9dhFgO+F2I5Qs?_UvvIFK<*Fj`I7<6G#my-%M8Lh@2xmN9r?$DGtMgvAd!hfICAyCbk1_)$W8UjXdhI7NJR*z?2mK#56jQzY##eo6bu8+X3n^!ky;
k2W>w&QH_s*kH*C$g}svVF58lhyEyBs1ljwr1L}PL(TYgfp9!5E1It&P{za1IiUs%D@eXrI_<~919>%h;u8i+NIge;m?lXE6tgFGjo4Ko7Usgmv`UH2n-ih6
iR_3NFq?aFQ4b>r{2O}yCctfk@tTc*1unc}74`nCiISuJCoV3iz{34h!YRra`1HScYTIanqrE<UimL$-(tR;`-{uW$y2724ge)+deK^EzK^4+PD(NcSQplh8
ux|L_0nrObJ>QR~pvW@PIx1cV)UC|&>6dN6spINmmFk6&0r3%GX9^hUH`QEDJ%Q$LFNWQdj^N;S!$)FEE>s6zON(w*#v;Z0p;-eSOp1Rhe{xqAcRow&9Lljm
<*>4Qr&<Wuima3;L--*r9;bF*PXSj|K5G4n<VO8w#g{3{IvDZMWW^|$f$!DPf2ZO|@WHYCl4OB9bXnI+W$ay%^+)Hx<5yznA&~e>LL7n$j)CdON_MQMoV}9&
L;<OuzXgcwg0Sym+k3f9O`LtUtz*{i3u=o~`V8Tw=&W3+dhwkfY)WLrL5iB#tqhD^iE%@#PRq!rcqJs=qbaO@!iGtkZ9!Eb2T%;xKC{iTMo13%%LU!NXi$4N
pF&;=H$qx&TKjo`&rr^Nu2mX=p_;2>YR)*p<HA{ZP74F-$Nc8y>HqWZzpU3IcWu>S!en==Q#b@QGR?7dIhOe2Rep%XEeP73r)6{dT(I9)o1;6`3Xh4LRHgTs
;%Bo7MQnmR>a;9Z<Rl!B&BrG8H-`ZG90j&nx(ndyz8Z9k-Vv>tEM#|f++lO|+Plc&R5)oIE_JAF#MxWdSC<ouQT)__d3YrkG%sJ<IXwwOjuQD*2S$H5=Zs4?
Y<1w2?77<mkpXB*$`fRLbP}%dr@cN@c_5)=xmj^D0z)h}Qwf+;;LJvK<=fX9$S!1dR9#I&uexVF-{My=uJk;Axatmfl5V4K!rJ)op5N!WV>!02F-G6~ehTVR
C0PXd?O1$RTJFv72PK6M4O2Xk&~pe;o>>coS70U^$%S+{Wsy^#j<&|%4CV_5%0005*}Q+cFa=wxPD<0XH}RySX!6CUlaN0mV-_s1=eJ5%Ixee@f=M{~Hp{I7
+)*g(SgH(zT{a)dfKm(`{;A$4WHW-`uh>z0ff$ekWv2_O<bm6zVr*OYCMdf;Z&kJxp*~mZR?ELwY$gBOB^4@0_P{!)sm@hgb+6Ou%bP>+58VUPwlQ#{bvNyq
evaFw)=yatOYr3_8R^-=LX>WBR$Jux;#xa(wAg$+g8qC^tGF764er--!bxdxicdPfVCM%X{w;UH`3!sxIBC(eH-1xUgPhQ{L{QNV{-SMnN1b2VW5ecD)GaIe
9WL`gcb9hZ>4)YRGCsKVH1i3(oPTC;2t`0t>#V{bQ(vsnO&d;fk7CE*0ZXp8*B<vj`1#c^42{wWKL`0zpuPVbVcj=9eCDfhDsNB0QgZ4QHnSA)ysWp|xt|ZY
&^DQik6Lh={Mu}gP$C2?kDK%ne#Gv#BZh3#rQi^HOFNM81dVLHvX+lRuvPkO<3dd;d<?kxPCMk{#^<;+?aeUgME^U0)&ShQMwgm#s|`zkYHCv5k|FqdIn9V5
8ri|yE8J1h_;s01n5-feyLbGV-Wd9WEVC&5Rz@lq;)2)dzqw&PcK`S6rF1a9Jga<>@*}3#yNNG)hC}PYwacb))mZ*%DRW?d8>D*TX>MH%hU&ha<Mi}72)F0U
Uh<Cv!MvxJ!oDQf|Gt*CFDDp$4Z|djr-E_etMKPLS&<M?^_KlW9188Wl@Ic_vT<p8JU;ZL&;NZ#d0l<9Q`{TDJT)eAxyg8Yt8Jn-F$@b=n0te(W02iw-~W#;
64$QNRy_LA2s!@`#y<qz(KAP4`R9or5?*M=e=YQd-npJLXAHAI%|$+X#?~Jn2V4)`VhY5~i0z)-vQ%`j`h{Jht^>pI_uf%Q198BkyM4RG8RbR?i6_G%@hW?J
I%PZndK8ul3Aa7KZ+~s%#)AlmF{%(n`g`o<IP#AQsli}5S5$WTm>Ec?>y;_wLU1zvrlJCa9m<`Z?bI%qpfq6rnT>i!T#cdH*T&+8WE#np{un2O`I&fiMEanF
nKPxFzyg#puNK<&c+FZZX2R@l5<WG}ej#>>g`?NB%iEbkT%CV(NM+jvP5snA+LUZS|4_3ke2-UG-l#Ml=+r<8=cRgA6)VuCDM<HU3*OU@OwNUO!SIgGW;gNk
!hN&&VuJk+2qP-F<t^rh1U8TNvUfZoZB=iWcQXu(OiR=ReL*;tIMB42?ShLsKPtXF^g*lDRncHJ57e@<G(}GNAW+}iG{`6bI>`)GTzkB^Pq4RO!^{VFDud{Y
UI*bES^Bj2XDgHx-(5W>WrO@VLw$>O1%%yq{973wh;G`fDVH)&<PJ|%Y#)%qwYGlOYgz$ltd}C%muCfPhE7+m-8j57-<D1$v4ckh{m~CG#t4t)GV?uR303A@
Kcb*W7>;__pPf(w2@SKhHDW=(cE^>5F%;(C4AZ(=TrurtPkLq79f?at1xs8>*w4o;GSqDXM=Ncep%2cmWsl)H@yH*~j-Al#{_O#+dE*ebqYiMpI(<te&KLJC
l-v3eJAupmjl$$@Q(WMy?>ZNvfl=9TzM&2eG*6V6SQWWoK>O)$I)QY=3CmWEUw1{b{<%vhA}w)obw?m2(+}y2Ij2fTtr2jPU;gC}f6NreXo;;z-~#z??yY`T
&<7OFvyTPfLfjPLyU+d*O(o`5uo6S-l@4N_7vZ>7xgZ^+x93BYL{0nd`ytKc*Fo<Wj=0ugYxwDiHcopC#Y-@|qN{tlBg`%k5pA0hE(uQXo#PmqUw1}g+FIpC
w<nB}?!3PL)Ch%1ISU^<ys;QvE2F^~hJ*TrYf3>XP!XDOi=_=ieOT-zFD)y4<-KxVk4OhE=X6iWj0C`GCRv;qac~aN=gT(Lfsk))%}H)stkI-xJ)et2yN8C<
&5amH5=UIf9d^fK7waI0)7BuU+gXtFZ3Wjj|EVQ8XLt`S=5}Xa#%S<E<H^q!7}{ue&IvYzi{E*QgEi@xJ$8UO!rKg%b3PmID_p=(L$>`QDI90c9FhH_V25;>
Zg;=4_BarDHjBR04I@k^%LoZ=;Glf0;#iFu`UjYJD^)^peI}?tH0U;x++{Z3kni<{774}2s!kYFNPT+yRw5`lUriVm@j@owbX)GK7a~-*4T%(dU=d4mne+$`
nijtIp*aP&CmzX0Pu_y3#ZIqMq%+jtolp;V^@Fy;gI3wIKCrIf<Egpmfec~GlS!5yAXnvN;8ckO@rMlm4^LCzPR1lIKQs&7OctMfUr)qWefaTg!wS0U?q5_M
gu#ga(%;@_cJM|iMk>~Og89bDeYAJ&Fm{(;|LJ@bdVb`3@vC~l_V6jCls*01dNX^9FGWM8K(h0sT?jN)p0$d<177yFD3#Y}AtEUDQ$e0LuG8PzSQUwY4}EP+
^ND24jZv>{M(SYmsfXZ!ZxK-TG+5!MPRE*Z`kw+ROVrAG40bmA;F!gqVejr_eEZdFk?CWH`HA1YT~h9NvFNv+ofrhVk%{})jlDtHp?P>^CI&@)qA$`G{_lOn
<nbgLBBpdq9jX=`I%1E`xdVPO_jOQ`k<)zbqa`jq$dln{a>LC+^%v{Kc96Ux$d~^j68wz0(}Oh0Pzye3n9m&!kDM&g(<j<N;U2nNAF6|(;hzM5wp`$C_s)XX
$`%=QF_Rm6dTgrB8e^1q1P8C2XA12FFq6c7pWo97Esd)*55G6=|Dhs3-w_5C@-I*CM;c?M>OI3WvjTXp=nqNyMx(@kr~2_;KkA)2w6$E7j-J1H`CdIb7}aK{
7E;lNx8B?lN6%uUy~@2s$Ylz5wu7Da?t)kmdpba(Dh~4T+SJdyK*W_J`ri9(VaG%DA?B?byt7i`-DkP*&z|u1p}l<mZ6q|LmCGHop9-85k89w8OOJE&6HX-V
mlpnVSqS%D?td`G;fU{R0!ox`qrjQ5v_LDRj}LJ&%MLN_5U@<p-8!TWhb2vEV`6nstG?^9aZ$h*Hwtm>DFvKze0RxEnGaQdIv<Tl%@GiMx1biXAp6t#=Yp*V
gw#U2dA_OxuYbH<;4?+b>CDQu4JG_*C1SWI>WYW@W_MJDJ)zvdb5p({8-vx0bw8I?(L|kX)5KK-@ixw#3=$s@)}<7`$u)-;gUhVHs2dJMgq#p(wSdL&1wB<^
Bm9ugI&zuW9WruXm{$!|v8m}|JUFib&%0SF^~RRqv-Uh)J!F7Fnm(Vcz41v^t@Y?tNP%4ECa<s<Cnj#hPmL1AgXhNXHIfr@c<iR@5vUsqX41!bQ!XMnH^a8<
DyxOMLc*W<XY}w%ygXqgQU&Lq=-tgdDv$6Sny{WsO^Dx~;gY$d0u#YY<3Y~q@SN(-=slwk<=5&9s^`_w=>Pa)`YSg?dhZXki`K>4$~<aDb$M7U+dpt@QpU~X
dH%K{eel>_A~%vV!-aBhg<ET~h?@TUsE=O_^77B~mdE%|#<k=qENug7Wv4Rl+gkry2a@5QB?qV;+;wJDHiMY?C#;?}g%<6Jr;KN{(dqZg&`?_yR2ggSvi!Pu
y!ifo53fF!Kewo6eL4ZpCWX{tcT;STB|Vd|biuLl-c?F*TC~=&NzGGfL-2_~-&qAC3`}GGz-3oNKQQyz^3Vr|_xRzQW4=hgplql&Z369Ya*|Vzlpr={eyq_^
4Py6H&CkWD{ioB11RM<lU8-<-T;w3qZ-Lzf>)SaIx^U9^;W3_}h@+&N73|&CAe{PPHcF<8c0%eRLIznlnO!Ffp3}hcW}`a$F-@?MYIsL$8l#i&Iy)JYDu!gQ
DlB^_LC}d|>uiKFI;6;TPMas9q^wP<uR#@IFKdpuX7Yo8iP%l;hzi!uCmcL|oDWpx1r$3aN~pb|-C#0Nh%9@q_Q6?OTsn}S6A)>NZ9hwvCmT-K?hUx+tdb46
hV_V7*F*8B<o54E0R{Yg@%sFTxFXbPELu4C>SB0sBc}3iGzJ%(cZ8{IAy%GgUutUxgTw~K1%ng}zQ4hy?sN(2fAWi;4cJ3nd)zGGQaCst_NO#nDTC&Tg`y|-
TtM)mn5T~11I$0P-|%@U;P1)$$kgUYte@o;E8XLyy4$4=?hO8*{(7mkD5P|cFTGl~seB=C&33lA@HqxonvNabp9R#~e2Iv*!5tk<1%^lmqz}~HrMT^a7AhC3
D<{X1?sIJ6Wv(igskKM!@>0;OyZGqI*8qHTdA5?LWDSQhQ<AU55qN%h+1XOb8A{P{`==dzaYN%IS6-AQb_dTaKG$@{xTCxj$E#30XW<w7P2-61dYSbjA;DPJ
v2^~Sat`f<o2E<qWU!(n6!iAC4g!}Fhe}H|prem%FArnvTW0ILVkry#N`~OdK3@oLSCJX}%iuT}af4k%I_fX~;kbUo2klv&skd9>5aN0L_$huT+~<6Hqf<W@
fAhtOuKn*n#R>n@H?uuZPEw(GBqbO&M_=Xqw#tMu=`F3og=FMNpAh=<GZQ0&x5Oz@ypUuU-e_xT36C-jiGrsgVEx+|-&qt15PhlnKxzuclLj2KzLyc8#`@Gs
!wCgHBki3g46y!oG>od&75DBfPVe7W3;pUKGKqo^Tn;)?$n4Jv(=+$)G6ra4rTdP-)MzBkZA;%2JheyMtko*5sxlmS88sBX2H~fr?m4CU7!2(k8p>j}MLb>H
YvVv05Ra8bMq68B={jxYhOr*JhRGD{cf4S8;)|H(xGP8^f6a9Ex<K6Ne8#3r1d`@=PY%h5K|K9|rD>A~I9p=sdTav0%i=DQ^xY12%@Y2KkHf*^${97x8G(PR
w^Xl4d*RkMq2Sl0I@o31pRLDYkJbw>9B6(?;%Cz1JD+^gFtn9Y`-s;Rbd=V^k~VTU7O5{Rw|BoeX9{B5k1>Pju)J)yoFQ`7FFYpQ@q<O(AF=Wm>gaVOlQ?-u
5~(ybq3enAAfPQ8jQ0VqZfPEfT+;%RB?p6Hstcx8zcsraZw0~g6#4$bSj-wL5hrnmqAZxned7K(Y*%|6TPF#CN?QP#@|+NKpHsf+a#R8-uXO<bV=bs?1)cd)
rwYkxriYWAZn)*)q{_W&2j_<lza9|qqPy8v@{6VtxVXyfP`Y<rhx5T}H5>^2^7D|TTO=wujg(soSzz>-ptAR_EUvVchw1r9<Ge$I^vUOfP&ATI4}8mkH=;dB
qkB1O=3(*t$1yt`pVP|>esuzpk!7c(gglYy{r0VuG&?@)y;{4j8;0kMf|vBoZSju8O)kx9T{&%{>#@I-AF}(cZ0BZ7Aj&SGOZJx^QeMUlrjOhqdnoWrD4Ps~
ZXcL7-ph63bRE}aJgm^Mdb48musI68OPEWFn!=CJCp3IO7l$gBcP?<~K|`NX{rZ+IlGHEjun(PphnT(7eN{UYJUSKMe_0&*@3+H?ET(bqn@aD4byH}I_o+Hv
HU-DfYhm^z1B^Tk{U|o+4+Fp5_VQ0s_+r{1ZC9iXlX-up077?6Ts-yF;&|eJ{(Zc+gOelI70)F|xuxp8Kq~MkmUE9EwRc(WvyU4fTEw-DXC((t6URQ6I{IPl
E-wq6Ni;SysYWT=FXCWJ{z!;mJZx@sQyjj>iin}Z6vKPI*%llT4s8?A&~31^yi>uWD!%@jRT;SaY2s*NwZgut*IUhR!;tT5|5fd16*kU{T2X&<hS()yBF7qE
P*tD26QE*;$^6uAc1CHGD`tzONyOt$V019im?sDuk6luj3I}_P|3x`(3kb{GgoIW)A&xnf;rNIpjHPTN^+Zh|6wtTL@Q($j?%t^2ymuTY^$xx-kOS=haq<{l
b%nw4p23(T6;ws*1+U1kqnrBiC+SmVIN$P*;-IcQj@_9gd>^0y`L0K!sv&$h+kf~uGX*z}i_3gk>2}4~W!ZhZ>M;;965`C;XAfrPvn=`3DrozMePhSX@L~6z
s?@MOp6n!0y}PE5{KOYl8zm+X5dJA4S-=U({+G#vOj4jE6z?B;uL=$0T863)ceGOM4}LDU$EyhkrwXhz!AO)xo=z{0n-U5Z1baTKO{;cBr^^o%%?HN^5spuJ
6K;e5lwnhJ`pt4dHe_h#yd~5OP^MT+<yd76R=FXk(|r~={KCJ+nLryQF&n=#ui3#()=+7UP!p%U?I!t8sNo_x)%%Bi8hA)3b!lY97>|zxUo@yNMqhtCscMEL
`bJ%U1jh>@JzXUA_gQxIU5@vD>E((H!b4s>A*#sW+!DHb*9BV_-c3m`2!eLbH<!>)3+*S4d?4E6Ue8(MxNaIPJUZCsl3i^AKYCBHmzQ)<leD^P=e2kIlM&ZF
Lp412&hP$GzsENP0aaz<YB-x;r!!dX3+ZFe<H+aSP~^Msp~RvJ*krm7j`_&LbykS_WRyCLG`l*U2Ux@Bu??Au1`whWAKBqz2UDh;_|H2Q$k?)$GAwq%tEF^v
K|M)G1xzvbG3tZ-;KhO>vPew+pzAJPvw*#f-qyC=35b7qmCHRYkB`4H;y-QaAYIFPXk*n9<=bu37&QgidD#cLTb_7cvqs%O?g_`5-lF24Oo&%34aybQ0&Cdm
Jf~9zh)-PWh<)t=wne%H(is(GZ0wL|w|ZfHyZKhAg8>eFKlfre)fU^~cEzTrtWdZzJWZsd`=3soyoI`qyq$smBbph7+W1<_{CbUA1NnzJOOFma?D0%L$Fzzz
raqoyXEpQ0JkR~&7?I<+%t=gl&`S%?O#Lps&Q%7P!_v<{BP+1{Q(Pnulfta**9y8GPV|_+@~6Hk4!2{zDRu_YSonC~<64^*h(whX=wAv#Ycjg_EtMgvY6CY%
=y^cyZEhVDVuT_0X<?10Nf7rr*FbY81S2=9p4`7>2&1bmaWPH$xaZH(Y`_$QFMqku)!q-my^@2cG&fc8tl&?Rn1vLa{ANVQW9_k%+)~zCU<j6--EoROXXFHa
KX+<V7vIk>+xQq6LA|~_-`FP!3HMXul@~o=D4cyEWFQ6-ne)WDLysZvw=BT)P6yMGp4pztTF~bE_=?q=8|=f!YHvJBLQu=XyS88z6lMEaX@2mB*2Lz)!PZ_F
7}FNCRq2C(Xzj(rjzC;%UGg*A?~UkBS0?89%)!n4WkBn*5|(a~<_|7t;l}TKZcRI}=p)H9%w*BU;ro5wPQ13b<7fGXUoi+<e-v0Em=xgc_b7&Qc8>?DWi#7l
oUvMXyXIGV7{IuDW;oCo>lFHT#$vV5%WdCi9ioGkuwGVcLm!;GO>yD1mpbOiH>y?N+Jg7F_ds*09EyT8{hIG-qE<e+VKz=0A(FA{-w5>}KXK}s9~nE8LmGt_
!!;58obJxWUVhXo93u{D^@XSH`MG&(2aGHdBwL^N1`GGcmI2{Co^raNo=F>uzbh;!Diw6$e9bMU|GFN2?iYUb$~q8~^xIcO27(ayq*nDtggF>0Khr7FMqp|7
%|)tTy3md^=p=|YK!lHX=*=t_%vq00D;e4$X8VIA-+cwVSLqV`$K;9qf=4SW`vMV_*-4!=FA4R(Q@oQI`WWjGt*+zqMHKx3k%Y26y*3rzw`--viRrTj<EN#N
P){yf;_3qS-NJH%dvb7=Ka=N|dK3xkULxvmJW#Dg+g{XdkGuyF5jKUulRLk5M(&wm+uK#9it+?F2AEd1s!t$!_YO_}T~#=fq}p!%vcTd~+Jv%ZN2C%B*8VHE
g~lDTpKrKX!P9lv^O2z&*5gDTx@@_?tbtQ~l3fP#oSoDPEc)R6TrtDux#w$BOJnawOwo2YMET8aWek1cN^X8`4OtPM?;^M4(8coa9D$S(G+4g<NT#-c;M)An
zrKoSWZR~WcoBfmmu99ZZQNKJn-!vUvO<V(ZI7z82@Hw<_%N=iV)tEwc{ZgKEEi<!?HGCSR=n=@sShepNZB7JFscfAqg(e$Z7ooh#99?|Dh;HzKhD4PwuP~Q
qzpB!A+`^TYo5RCju!{(_FW{?1l>}85@{z7Xloi~+m37Fa9Nlt%O1yDBvj=5jF3dk$xStNG7AvfGx{`@XrbU`e(aGFQ}}a}easlrMBJ&3inAS|$b5R#%WRAa
&Xa^y1P<xAd(y{K&z2V!W`CW2?B!yi51db*1fD?X^6~j2b}|@`9DCVf#E;wU9Rqr!=5S{@adzRMEga@8tG+###khpP+EdqH?7w<>o>iR#gyA`-YJcirkdf8p
Sb7jzt6q;syeGy6n^rBajvIz_-9IMK7~*R4Z-+a1LJ(jSzskSrf{D*<#hgOCAbdVWN7`==3!j)lLP>3`e>A&3uqcDOgHPLkPDnuFu1iMtUmdg@Oz7IJQ^sGW
f#JtjLh+*CW?CwW126k_+&*wxBSJ{D!o)xw3bqHkoL)P@p+NY{*$NM+ss(1pxv1~ub-%^r76tISe{0*S*TCcU46glclJI{#J1WnoiL?I_=V}=|aOUEdRf+Te
_uYp!bC|52uq!5e!)ZPrJcp16&s@nv_Be3lo|4`PF&NWGl1>p@fQz@cLz&zF9L?o{jV)n_tCTh+`jUjndub_ocY<*yFi=k5r6@!%bZs=YC*VLy@gI}*GDz#>
xz5z*BF%ArEMX@SvCpR^GI;_J{ycZLLAV&ZeO{l062f55yX-@`lY~c~ho%N-OW^e0Ai#J$6=~d*i~|8-*ckZg+443H5mJ=0GkI0GLvZ3<gz4V+$(<xGehEMz
z3~B=$B~$kKP=S2SdZSI2kUpJ^+74xS!FaEgTwFrXH*JxVeaO7c>TUNjH^NSMmPlJ3f^kod%5@p$1G=ijuocjRfH6Re6hMoEV};36UO7=Z}|VWPJH7wWxl-!
9?6htd-kZJC+&wCsZSyLRpy!*Eh3<$bF1lfZYsQ}Xj0COrC>DiCh^vSA#N9YB&1!5K;qP*N5La62ygI9<hBLkwijE+$X^-wiuET2b%(&5JF5R%sy@oTO%$*U
`$6iUsMcIaI0io+|7$au2TSOUIDQjD#Zt7Wi+38jg86R;k2*pxY|ZH7T`yR5Mp&!ID&i9}Prs9O*k0ax6a43NG?wLz93{WnAj<j4r$f1+h_kZvracyoC*0n3
e(u@eE8zP`{L&9Y-bbet<zjJFK7MqoJrUZU1`R?!mw@w%-2McSIE*{HjpcQmMEbd(Di`wHFqc1O)jw^HNXFpQ42}@URf`Q@Jha!Jx%N-hS0*Dkj&*;FnGK{0
_tS7wB!DThoy(xe2g05CnY{|9;mgYXY^cBuX3a55vz{qn?IU=)C>xI7q#U*+f3k4TFLabNSPPD~LoA;bcwi{D`FIth9jv})5Z#vufwR(lj{R{5kdLRU)HFKd
n~Z_-@=O|L{BCQ1x!{eHip<}&*<BEG`0kd#=SaBqS2y}A`h!;`O~o}n1FjW64+T8)00&>JIoYr|HrNftB{oAKAAM^iaIc;(>@(qfA`*y}{6k%OkNnWXzEDze
%^uuN{D&^<1;Tu-EPnK%D`+D>aX%l6h5T?$e59}iXcq1#)ud^ol%eV#KSLk_uD+rn8_z(%we6N0_fEr8^VY~NT^f{iQmW|o*CFI=e7M}Z6qGbv8mc_&it)=7
5g&uR!Rn;36?`f3Kb>}j{50;{(|hX;{c{&D#$%i*u#VZ<AAMw%!X&|7*gi_F7EkMrZ9BaP<Gs4+pm=}Vioy`5BCQfF*+L<oWvxH$>4lR;0duc?i@{tVmQvXe
41&4_jV1#}MDVwmODWsp^!Kl{1y^H1A<{4!%ohu%$qCY;89%g8&{rhy@y7=ixrtX9zSyTMJL^ed4-SnXenHVf4Br)6o9-$@-ng&14p|4N+)gW9SI7mca-ATz
NFF$IHpR$za^V(vv{p$k95#c*ZG*oek!%@ADW_eG$t&!Zr<L1b)EFgkUt<t&D%aVJ$*Zt!vh#X5w*?a~MwRjkBQQ1*(>nb=3ttVR=4Zpwa8*jqe#oQ*M)XVf
odeFIE5)3itD+UDrwzD`kL=AWDQETl)=Zpk8)<#x(2JM-mA+MZrEqOptK@r|0aY!(d;4CbfZiuoT*Dv+qmCcO`=bgW>?OjYs9l3qh8sp%v<c`-nD*|SyagYD
@p^TK0uXI0HXDT}qq$|^Q2WPP<ow~dP5rGBu93_nRwQvacr}yFW#baO&Rl+*)|HE7u|-mmA9t}@-rJrPoq@(qvBG+aT!dPwGhGbn!eC>%+-m*>xc-{DV>{sl
>)qai9?x!|q;!JcwlW-A-<>?h3nFlLz2eXGnOu|@y6JG{=fX$pMSkz=2+Z-kW;gaeiz5-q1#XciapN18aawQ*eyrU#@TKa7wSIMH%DHUx92`mPEv<nJxrCzp
-uQ!r9lr>l3PkAe)@Fr34z%NxX}*xv!Yo#}V$wJd-1e_;I&ob=>~GqNIk606%^I|g@)aOqj=)Ofa5YvEw7t6br9k;P)v?yo6-Y|;Pu}!R#~*Ie-(jYycwp=m
^2{|8Qs-ZjbRrE;KKuwu6uXL^3|YtF>}vRicyRHyRN(3GX95X<ATY$}_~>4U1&hQR>Fu|Lu=q{E6Oo#OY1S({e+#{^{|sgLtJmkj#eGPd`cNDC3w}*V1joS3
?az<Y(hDG65WRLYJsQo0>X+wVr=acYvxr-B#W3DzT8lF;0*lRV$Ti+Llxk-!GhZLVi>$BI-P<SO%JBN@=(96eY%J~?@w$LlyyvxDl5%iiEaxi8L=j?+J(Av$
%!BHg^t^+WQ5ZWRk-o@T0Cl#byIj-dxZ$iUp68YbE4d?Qe?O}N$t^KEdzrh?kV&z3Q;ovwiR0t)RjHs(j`^fGWP$uo0#}@3BC+2ywa9hX2U2!ri3!ZHc+^o{
B=o}&t)G6Dr2jU;!=pNek2s>?DkhWg$TAP=-gF9=JS$PxsI&RBFa`YE{jw#5^$-ZozAkY27LGM&DNR-7;E=_i$Me^V@cLtun=5$$+NBL?s4hmK;6#wg1GyC3
&Xd-RdhP<LfBe*XwHNR}<w3bwX%;RiDZ~c7_rybwu;F^DlZcu4z4jr#8l1mgr2H{3#HPTv7th5*aPFD->2HDNpq5_JH>EIxJp1*QgspI#zjA$zN!|sKAK$SE
OPb-5L@v#}%lAMZFx}8rb_tiG;{{gS_x$bay4NS(6jc4a5HY!Fj;X-RH)rV2<J@;fiFV@CczcHL<&E1K=sL$jEJjiWyLW}wTaT+iM|ZuucEuB=hn#c+E>s}1
xRcs%(j0Oln>(gjQ6Tg>FHO;Y1GQhY%86f`#H?Aamsw*FWb5cj%iT^vm8-9GY1tMjx#wT!8786q$*^tfavYXC58TZ43PbCW6%EpL8_Yj+oV#tE1b;!HvX0er
cr~c@%Zoh)t@0-eN?XHmxFA&IMM)t_M``_Mz3ecrvCdsPmyGCJ&e`d_@rVu^if~zV{m;M67hZP_goT06X5VbVnL<2p%~t&~kb$w^xh|isq#(THqP)CW176n!
{wpPJhs|oR6S=@ygr$ewwp#SZg#%_qMf_*>d_`kSgi{??T#t6V>ajqbpAnPKNFY*mE2~0evypU}l)y{b3O<g5C86K!AzHn1ezy7&#-^#x)1Gujq7BJO1(|Tf
jG46&33y=ZriqO#wG#}6SmyXuJitiEIAZA@i~@sUPg>4MG#J_`e-Cp*pX`RT9*n_zx6N4KmK_o(T{n}I)1b>SQn$Z!FXsr;zHfW21*(I!M=jjt(Nlbp$LpRg
nnZOnc@!K${K&@skg+Saloy0apoGXAXI)Qbdz>Tony-H54EiFWzbTpl*!g!uw`kY|c{vR)!jieLID)Our!_DX(D|-z%@spz{5|6ek$b#xv$T5I7r_jnhgJ^C
L#HhKA(4^`E)JH7KTLLo*|#bg)+dR08noZ8C6NP0#VfITHqmgdm3BDrRuVnll%MV#wT6fO+xLxCE+9XzQfTTC3N`-G{{0s?QT(Il&b*l|D4dfT4M`24-1ke}
wZa_VYU3A+cBBze@P4kJ*$2M%2jjb)Jn-w+^WorUGH4a7*$E4Ah5h@0OGozVs-@k>ZL!=Na?;1YZW37|K&v_aXmAL=ONQUoIUE781TLeetl>y{@SDB5+yz7W
I`2XqoRDh2bv~;}9up>JXVzv#!B6%$%lxGmh<CNj>WBi-NKz20*Kdc3ol_q7{^>v^agp}HetQ^le#k0%Wd%caqj>)=FEmM0?5|%}!Q)r-HpKA7_py?aZ8ukh
dmoyp4=qCd(0i4K2LREmIo4~L0r=eDFlfA{jY|=cSx&Fb@NHY%QfE~kSJZwc-`&&#Lsn?RxS|)bU1<(<Mw&vwv*fKVjWb3<xPBkE^1_DjIIZ|gA58i1$i<7;
qL+UD9bT!zDq!fFfUYh^&$pRPO4))`_4m^n&Ar@rn)jUD7d<q-Fd@0Alni3$Ck2<U>%*~2=b8D8HafcJiB11W!|C|XVKHr2Y^!kvD7fn5<I7KZ4CRt=yxjNc
=8`c8qZ-~hGKJyzLcLJ?nhz}6f9P<HxnqFMe(=rLIK2Do6r9xJjf?hI=yUxHLA277m+;*Nw=%cO230No(<yPDn}y|;F52xG<chv|;YSRA4UMZ2-uJCG&x<-j
_eb<jOsf$rsBYa8pmzpczBnuUvI?H;5{8^UqK(@ytZg+{!&c!D>)f3v)M<AToP;Tg2?xf@9=n6++7PLMi9A9+gpXKV^~Tx|85JX|H^NDD&q<|Q;AhwdNy?BV
_BC&6o&0A4;=+oNnMEm-3Nx*nCFUWnlx(bSBn3Sr`l)oZ@vz(P!^W?e1q#MrEQh;`kP!8(g>29YM)Ci8!cue*OBv%KbkQF=S8vK$MuvjlY4k~$-ets}Xwi#M
PsQieujV1ykyr}-bvyJ+Ivy=PjZ$XtK}*Bm8xc8?;O{?5xaQ%B18*JaGezoQOrk)gS-)3DKEd%H>0-g_M8okW$pRU_rve~R0>WC=<7sTs_<T02!X-TgdY?W%
T8#`tv2Rq72v;Jur}o))UJHd|*jdLKkzBM=uXq&8?VYzIVSaPn1Jo??2kNVe@sW~9hp;mgL>~k)KaZ5)mTpt_gY+EKgm<02_BjIAC9@K<sqEnRUa2K~Hv;pg
L*oeZ;*tJ1grMw~FPyv2POW!E!97WcL!NPuYqJD)2~-m?!xqHoPF9WfqBCha4?{3Xz7&-6%MUgGkBIYr2lD;fxR6x{X{pE_iO7mN?Y;Njd+)vX%-&nb9)*y|
NJE3nCQ3?55tULN-(Nn@KX4z%^}eoiypHR>Z)1wB)edbOB6JEJkhj9%Hxc1;gx)x1p%FLqIUdYq9QD(eqrnz;?D)u=M(F)8&3t@I2h}eukA2?jrF!?8*gWNZ
@SD(6EU5VkvbijbBFEBjI`6h1w^<CD%jgeXm^8-%r_h({*WHn%<aN4q(hG`Q9)y-J67iYwub@MS9XPa@d}VhVP;$}s=%2t8c&+tr-|voqyr~my)8ha%>b*E%
KIDsUDRYb>nhuDzmz>vr5{RG&-2TB{?g;UFUZ42W1KNpSjZV+|V6D)fOUT;;a;t-5?UH#Y4B2<)Nsb;w{I*P=JT=0QxKCR~Wf-E<NAKTCh{4)e&c9`P4G5o?
BlZ>v1~HfKA9)@dNEK7&?;j07!3o7}HKIyn$I)l8Trh$BQFh8HA7>Ew-fwO7i-Lvfv=Ga!IP}Ys&p4)<<MX4l73nfw2qk6s^fuK7i(`SB`(nIcGam55H_{Wt
;TL9eXbd2b_9y2MRSYT)UUL7o$87-%CS?srO_3|_&Jv&LvUe_ba*cfoa5HS;Rk)xbmdW&J{dZy@4jw_{!T|8bWeU+gkwbkSg9^=>6V8$JL_}Ll!pIztH}`bl
o_TDw)=mkp^q)MumSv0m1FH882NiHpjNQvFK^_y^*H_|ix`9nHONel<r#+BR=M$0l2G=SX_5Mf?F!6C@o}+h0Oo+0;FQg*frY?`K-U03V?u6Y-567giYV}~M
8Xo<vU0o@10^zT<TVw9Aph%s$Po;MW4Wz#*5{lfxpmNEBoZAV8vHg=O3F06&Cf?Yl%Rp3Jd()57WY~ul+Yu;Ap&-ZhnBRmYWHw&;>DH@*OT$$y;<ywXI7j1*
jwpk1-{!A)0wYXjO?~+~qksZQ+6?9w$@uk@+M~=e7^h!15x=%qMp{T|W<P-y$cB}hJ!H;fT$h7Ggw6^2#-lzG7Xz@%v>WyD_ZfWHVDuo@EP!LpSwa7fEEJaA
UfWtwfQuX1fd1KF9L=%r*x8arrbCq9A7(u;tJC$?=-WahtW|c_$PK@XQaT$4b+Ds!s{0(7Ayy+Mt;YB?L8@xCMmXpVD}}Wq+IMqtL2<~bj>j4%%LQKi--Ym4
`?f~6jSpVVzq&ZR<PR^Q@B>j<iSX1u(fNDM0r{tO!zh{@AnTml`T2kXE@>^rP2G0F=Y4<j#WRdB`XjN(CjI~a>yS1WCF~w$g12{~)tc>+KvKG9^`gQPpBUCm
PoDAxnI;3DH>n?F{^SSv5u_piWq1i?fi3p&>oSlPx}jsT>x4vEJjPtUk~S6a!$#)bi$f!d__)q?yx2nq4(-mTb&tw}DWk;d&>?9E3iq#WHthMo2$N@0yDw;}
r~ZmCd4P=OxUQk1Ixf<4_tmrLV@3YRFinOWPCdI*CFUxKW=7dZwU6x)m?qW4T;++~GVjr0V{iPidRy4SehDc9cNZDrM6vwXXR3bF9>+_`w3E0l;!BxxQ^?Q5
a2XmCd)Q5b<`YldWUmQ>nL1K9Xpi@GCnR>I;?6_<$H%6PgLLq%x<H;=;sx8dPMwKQdpWl&Mw9Z79~Xa2t6ei;Mr4XcZfz7T;?Aed#5C&QmWYx?Te=k_y?@<H
lD!D0j(rlh`_7|!oBg<uu{1>DtyrUJE+C|p>SnyaWwc6vrK>IFfUVk^{c0OMUM?6unG+WT;VECj;?IEJV=kp&6)h~cuxONPUI6jz7AHxWGDvlb{P)o_gQ-4G
?u&#jC^gwn|IF6J@I$sNGj|Ux&-E49dCK802YKt*l^9%oNcqh;_!4Zty!c^IAOq6u`z}RGmvPIEk7`Cs7~CCk7bxz_B7SPxFJ@2?jxS{s_nV1;Ld)o4)GuM!
Z}9CK`J;faRIZnm9+yy~=|$)lDvj2TOFCmOjw3fot-)%M2^(~DK85ZEa7eG^f4Ia8D{b~P_fP|H7mM%f6+8+5l?1*OM^OZY?^B(bkc5T>uVqG?5K2A?FTOa;
iys$ToO**Kp?i~bwWd=W`K?EFO74kcv0X6#>UCaB=APjHsCI6TN21Ee?#Y1suJP@}5-Ercy^#<DH~33<1KC`K(b(p}K5Zt51UAuiorud=*wvP(vAu{KpUBY8
RA6h>kZbV+GZO61rQXPNgyTDX_N3nPC}+04e)8l2JYTC=xvs1XDkr~R74`BEC+f>FKP-md<mRdLm(SoZV@BEa(~?NcqrD`)?-Jhf%|FOmK98ch{-;-m6=6bn
>I!Y77OG>!&CeVmgwZwGJ82<e7+GL?%5Xpm{7R;Nk>Xm|_(#`nMIa1w-AKurICGT8%<=O~OGAkL<Huble%!g=zfW3Q5MS*JJXdcD{ioB?pZ0Mp`qH>Z|I;qj
Q4Q&0;^EcgfWgBq9obA?=seV-Hj5F%B8mCQC^kvZ-H<wI)Ww2?v;D-Ex5W@iE^vV(oD+O1B0;P&l4z|f*w*6{1bYV8W(VsL%oE?&%Nb?If_l4<ag-ar-Q|7v
XZ8|A#J=BodW;E6-a?7d&xBz^DLeBx>?l01eM_j}7DCxw>!Z6R0pLlA)tNr4iR@W(sh}|hRND;2T=}Dn^4C`t%xVHaK~c{`xZe=&dJn0@B>53Wm-=dpn-9~M
$Tc5^Yhp6xIZNed9mKPL_IqothFt}k<8Rn>!R}1=YvH>tj$c`F8nXz3N@7Wc+(BJb-H0os8?uC?Z+rZa7Ef4IT^V_*CkK-!b=$peCMc)uBCbg1Kyt8|z93&1
8i*{;89h?KtA`UBf0yhL{iHU9KieK9Nse;|L=};G^Ct`AElczSFZqmK_Xb}YRm+VpDp)J&QO%sy#KZB&Eu$Xt_-Xu(n=3yQbmM6}pM0euzlo%tYj)T$d{Fm=
IS^$tl=7T>mq5K0!*s;k9u|s`4@wIqp?ye#PX3lLvhT#w5<ikd#odDM?Ekbdb~>b}`;!&~9@gE^J8F&4Z%pQ9kL7XsUPe!Pnh8$v3Y3;jGvV)3%jUjIBB1th
GU#XE#ekZ78`-n-SiDxfdA@-SZ;vs>n;g}I=DTLc+onu7Jh)G=;)Wx(W&%D-ezk*oy^4sJjt%nYLdYrh8-VY(<2QO%Z^Tg^=?`|2!Kc*Mx(QS&aALZN&pcKz
t*R_oOLE4uf+49!dwX!@1!&eBcf*sL`>W{=s6$Yz^3sIA2E_g?y!t3+i4?`%jYl<BXj;zIjTRCEC#~jU#C2m-CA8n&NLPcnN#Ds3J4Hw&Jg!?HQi00P&Og7d
`|ahjbg)jHCa&4g#?GHR1mf48z4S{La4&jUg6q8=v@iN~vOg2U``Tw#2k$duH0okh(^>$?b!QJfAU53NF$tGrY!YyHKK~?TO9$CK?H>9M1)y3^ZIWFrve!FD
Ket|#g&dK0u8f)*YR=h5t=^EtS>1k(+$|e8oW*MRDkH`f&Ng)2)Id_Nzz4cLPE9_UzCSun2+7LnagG->aV!mzIx{x7%~l#zarh#Hz02tP--)6xn9;3K!3ZV$
S(|8Cc;G(SQ@oXDip(B{>aKVhICa`&Y*jHsXCZ%EInfzUjU40+x2drm8<TwclM1ZP?pRa1D&awMS!ty!KO((+cYcqY#^R5+>G8z$_|jNkf5TA#6_=YtM#3&Z
T|Ld^Zln?(8mHO3Y6(Ro8A%2chXUBi=0ut2?T}Suw^gvGLt=qbj;l)-v2^z>zCC2XnaSIH&q&pfmU!<ziK8u|U&QwM2Qs79urFHppC&TsUd@r;RYXM)clNnn
X&9Ae7E%RB!r3x3sqCsP1W$SAoTlNyF3;v$DkpzT9Is;EtT>4N&DPtmbvW_2_oSj;8}l9*=fn&SpTsd;p_EWL3V_~`nmj=pn*o94_FYO~+Ljhrx8}w{Z&Kk~
xu#IRPb*ifOM};A>-It;Dq!O?BxSj$2j$cBx1OK*bou$2MXTqREioQq^nv%eCaR{SD|_A9p}q2N!p}+{Tu+?@`~ps(I5AUWln)>$tS+IbP{gm9-&6&j!gwCd
BJp`n1h^V@?{u{=TCSbRen2jPw||3{zt9@NskZWO%2_siAUjULazkg2lZhn>ia6lSST%Fvg%Vz{E<`Pl$soUdWj#hk0CB=00`qx-h-vLl5INleuM<;LqH-Jv
t{zc$zI6#KDJHLY12phu^jXVtf#iSwJ-Ay|>vCEX4_MOp)!!xHL?&xnk$?caRno4dbe@Nul7rT;iwg$X;wZVtjqp5CJ-%Hk0Nk~wUTHk?M`B>Z#NEU0cz7qw
?hrE*0wyU>IT?B2i3VZwOph&O<KFwXs~6y8PF~-FsxMBltet(tZ3CUK@fO)Odw2}isz)CU2g#j%bd^&sAbq25P#~HLMl+5_<qv%j;yqj%xaYrHr>cxOa=j6q
mT!E8#|y3|p`u%(ipVtnYNi}%iHTZT%kN*baVoC*ZR2DVK3K)K22XO}Uhkhfw>C?$L{s|7HJT2*+BXI|j#}(-Vr*$vRt%10?Wp&KTH=q$gyx@EXGprg)JWLX
g>2C~#<`2SAQ}=K^c^w8?BZ_nPabbrs<KxXmjP<VNu$EnUNCBNZaaEFA8*(_xLTF<Kz)Fqmdr>M!&*u4GkbH+;x4-Oc*+R&Jt<ocewn~W<M~e|F)fHaA1v=S
lfh=nzh57X!x5Yi;%jFr4ble@dB3tOp*obIxl--~wi)*kmUU|!6OK6B8gK^HRP=v+Xj5TS!!D>Xrj6v}iz&yVt*~E+d@O-W3@^KCbdB$s!qU6zi>kdHS~&eZ
7B}@kxDeqOIj;$$xOux*Rp!udG##5uG{oS`)++VS<~U=zaUf;I4o)E=$s=S|2!3zTP#o!x)+1xo&ZU>}CpF>24qpInsoimD^>+qO+cH0ak3W{VL#%Z*t>AAi
pmF-JI!LZkU6Tm##&g|jmro0pxO3&}Xmge#o@bl7jGnc^thmY6z!i6>=6s+ozjP96@(xU$R}7G0JbI4ynJ8!~vP^$zL_p@-FufKRAHod`BbOsIVKvc7eBMJG
5ufId|8uj%i???iIQDww+gv;6&bK;v$TMUf_eLM%+(SJ<=59!MGC9|&A&fdY9T6Q$D{zu(9eY`<3B}`QvKZ2%u=%Hn>&}rv=&%Xb$cRWlF?@h3MA8Rjk@F${
yI(z$A6vIz5{a2y`Rey+O5h<qw=1GzjD|=as{{RV7*Y+i<Q3P!<=4^~g=z}8!l%0NI)DcaEP@#=R;nOqzaUPpy2tS}Q!>##8hE)R;o(gjgoO|q!#)`+kZ_KC
S+vZ;?@AN3bax$e$?o#JlW@Sfe}wd%>Mrn}mYB&MR{Kw<?3Ua<Y--Mk{YqNPL+Xip%O|Uz7a1UvR;gmXUloTx9ZN{LWq=$*Cz~`19T*%-^{803!N*d)M%MR*
sAY}#QzGS$SH}ZPXw)oWG<f%!#BpmdnV<Chw=RgoNn51q?@ag3(fX+(GF#|vMc=)~U<gK{?78V)Jv3RpzjZ1*1M7{iJE;_0QJBv>KsBKOiRtAVM??ZqU2`Gr
ex(;;i(ik6C0gR!32imM7CpQ@vbMt@>x+67dUgwDH}q%<5)hVUg7NWkf^2F(>}D2c$W%1pIhr4-Gj;_(WEM_-({#o{=Gdih3QL5x^gh>Z*vn%{%ZtR3MzCwL
xtph)jf|RkaRpOrWY90lX#LZ}DWzwhZ~gMaZ-wD?`_%#rk+r3^Y*!%laIM_^zjk<Ts8!zS5RO-^mjk%w?NPBi!#;b}1P3S0Z5XJv!0?T{U3G7r=a)nL)4w~y
PUq9OYC<ranG>z-k&E|se%6^f&M>}ox7@H&4%(cinZH}iaV`J*L$>~4{8F8AF=vlK>zFXz;~Tl4I%xhe`EU^Gb&quRPz6Ke5~Z+_OgbKKNX^zqTY#znRb+)+
@RcFhH-0$)Bg0CJ@$UjL@a*&Cmj*6OJtu#%5^ap0z?Rg{E$Yx3j#%h#vIU8OkH#^>Vtk+I<^Q%?1(eAsM3v^^>WG<t>qrXJucsX|+9^Wp!TalM{keD<o6Z<L
o(i_xBoCX;>cP?eb<?b!C(^%#5_J8ufYDK9A=L;a*#G8~w+yvHqOut~M~)AcWe7Kk@|=;A`lI@*b__Oxwgg1xOmMH^qx`IiDn^N$w!YDZgIJ3{(uytu$GZa^
4_x#|ja@g5@y9kOo(<13OLIiOUTUwQj}87w@sQMSXn=JpHqhgm8$4b=S}vrDL{6jRM!+LSn4DP=viRMIUzhKVT2CcIaG%SSwzCe1KXl{hdb~AG3re_zjhDcW
wJwW6#S<pIt?~*zUTDkGmu#XhhTKP~?2RwJkf$>3Z}<}r_8;T%`~mft<IstFdmtVBk(z8h(SG>mP<iT5swVPCn0$5Dy)fmz@}5j050(NvTOHokaKC!!B=d?d
uBA2cIj`wqH#w_WKiCpyf6X8J_fHr6-zW<!&hbOl(}w4jf)BpEUWyhvV2$JPJ0BUxGr`;_d-&j<&QcE0C#61!!}r^7y;4_wQRn;jI6;{`{FvW6MH?ERWT}>4
bIAhShpX;t2YBE^v(EV891XZAzG!#4bOPHd$%S3I{BW2py~x$BgZEv!ycag}F?pu`>R(nhgy~1}R^HV^`fKgJnuC&Xc&7h$Vki_O-_oQBUg+ZyVaBCc&0zQ`
*4KobNk@Ll`!&M_6HJ^cNOqFeN2hp5{UpMXV{OX7>F5A=nv3~yC0V#MKFATJErN&1C9kTF8DgI@(M)-~Hy$p%jOgh&1D4x`1dJnk_*VKuz13VCk$P*_OIKxZ
r=y5GGhP$9Vn5?tj6{*LruqH(f;YUypQP=Mx*+n>P`7fHHv$$6&s0+>Agi3E>(Fg|l(Nf_eG#+9`Cs+VJ<sHz>?~<fw!aGKM}D(VMhZcGNaSN6sUQ5nxsmPV
3^MnJ=7~~fAU|^0OXH#!jyc`$@(s|!zP!tK_&2@5+sZBV;j|AL+ID}4osh&G#e9OFicv_AnR|9?+7?L)=?9c{l(FOTLM;>~P{|*-vima@MLRc&f4wpUGh2RL
X}k`4)cl%~3gw~XQ&M&6Lkt#kC(;EKY{2u4M6bG05<3y12M@P|L%;T+Bb(n|t`f@V-5a(-sdDF9)jkcJd`xxSiTc=o{@qVBl~q7wjX!d%QeM1lcpp7H&1tF(
58=G(VXJgV5^F9Cr^jNHuF^Zm#SCh+dQ`%Rj$oCi<y*O>31xooea=(?Xcmfet1V^+<B0*i#k^=pO@xt5Y23g)cfQJ$Unw{oLXfVx;t#*?p^9t<Hn`|P|JTzi
2CjjoewOuKkiRZnw`LTHKEr0)Im<F+mj&Y1AwP&R{Y;$A@IzahXN*_42Si5uX;iL9;kdK<pI<wM=#mOZxyZW5TSm@~C)4~faD&&v;z0&D%zpiG?UsY?`-&&R
M*?t=ZeI}lLkT#^9=*A7&J(G#8^5^iy^w6tOxga;4?Ywwo9e7qC>bwPGibL11<!}nC1WetKJ%Z7+P1@AF18ctk#2}T^TA}84p@D`C@dH0gj1Z;ejntV5XEzb
QK>Hi9o!aYQq~eM9{Kh34Gt-siO8B7q7H^oB;A#HB~QdUd)w$WxTEd4yT!L@aj;cxwOzR6i>W^Yd=XhHm~HtOZs}r+nB_-95eFinfB0rZn{Xz^!^B0in%QA6
XY|!mH4*{wjcSQ$`nY>5*4y!oDZ~hOL;UygL)l<5pu))&S6@gU(Af5d#sT51iSw#>nv;9AEyy4H>^i<a9PvX5QAr3#Xe8)L6>HRD9Z`GqH(@Ag9O&aDn`XwM
QPOJS?fIh+qy}Omu8a1Nax`5JwhqKnOADD<g+6`|x2A@h7~_{{Cx?l=E!?Y2I^#Ee5wVfpL3zRhk0}E4{`dXeghUCQ(0eb0HRYvv=ewZ(MD88J(-~OM(kS_u
rHn9*#?9L%36R_0z;>9x9@_5&_&gGm(8u{KmpSt?)^G1WocYud4t8FGme%^HQ<eOoaKHwUZw&VZT(N+^l34E#QwvBOS^V*DD+y6I?pyKb*unFGrl|h92{^Xj
hSTMHBk`0E{i%C<z7i};$6XVKQ;`kbYV9G|-)=?OEN+SUKgZ0DNciKWbmu_!k0_M(D7tx4`QvT(ew~^|PpELuD4svEr_1Th7^gcb;4vtZV|uR!)nmiUPk*{V
@@7bzD4z<LnBVM1x#&S%|I--(riT!g*_Yh75Cpf)aw31S0MzBM7_+X#!NRFw`#Hl^WQ16bbX*F7SGoAU`Yn5KP#&Bm`t6C9yDS7UPkjH=>6Q0-CexM9==QZR
dS~tn3fdM{1~nT<sp_!t&ibN_uEI^=uMSd;3f_B)dxBHvSI5|}0n~lNHlkXhkp3f-Q9#`b;U5;xsUvHzH?B1s1qa~lVO_R2ET)JTU;o6x+zz|9w?%jQ9q=zG
J%Xb(3=+Rnshr~iu^8}lGUS~z+SgiS!tyh~kf48VXWyO=v9-}C82EwOCOTSaw+xomRXW*hr8x7scVuNg9<NlYJO<zTAiuxGe&3^LShLKCMLmu|Wo+~knL;d7
UwrlOUAqGEG~a=LgH<3OJO51nVJZHJE>fuF=Hv3icgzG=hw<axGi$?7HZYHm%(RgyKyN;UryGqwK1NTo=5NL0FN<5ObW;|Z6O4z8S@PgRB@wpPoCFh(&dZ#E
36RuNtvpdw4^?d~ljJ{Hc=TpMLvuPBCU5-S)}Qx4>2A$P(5e?;$b3RbCl>b7GcNfgg(wcWkhJx`_bu;f9RGD7ALgF@3kuPRI6#~>rQDu~0a7E6rmhyaJfE&Q
-j)k*z8Sa2({WJzQ!X?AC=+bQvevi`WkI;SinHoa4wOAlO0B$4z<O$pIia>0#NVI)r!AHbC60Mp?Hko-F-`5BxGD#|fCukc?`7jA=?4aF%>ZO<&oJc>`rxI(
=Ig5jg$S0+KjtKniF|7I*k?>JP@k(S-L1+1KWpkurNn3i9jAU^+#Ziql}At9l}hj;d%+_-G6B4CLk-2ksi2q96nc9l09~zOE&2>PxEOEr!{Kr$w%nGwIPz1W
{UPDp=9O3siP^<(=$62`T0Z+zg&(+7BB3OffCq_d%pA0(XrH!WZPbkeNzTZAi00#+PF>7NvUEr{pSeaY=Z)4d_FI~^5fDx&AkSBd1y?u!GY-FeNWYWoaNFbW
X}3AAdv_~Q6!d1n{cRNPs{9OE*<8feI}dvA+Gl~X%Wn1Nqb9WWab(C*c)~#V{+%M9c+Am{*t;|C`QtD-rNekKxb1FA`A1e@?gg!Fh-?B5?*Goz)YymMPcw`Q
MT1b(-u)4om4wBwj^dM^@hJP`V&<|?2J@FH)qJP(@sW;Mzo0b^TDyL${-akgGEH%DemD~&oob~N^GSFbvLiJ4I|%VQT@=at)b@16fA-=^A=(s{d8M@qaQ4Sw
e9p}Tya|+Ab9+&SUP9yjcB(~qqcwC#l{X24Z$`{VBw|oQ%$ThmV2j<uIfER7HrSczm9Gjh#ABc2AFO2FsGl(Z?am^J+Yi2d`ATL4W>)=>?N5;~;Bgy{;;ckd
BLzd_#~3{F3Eyu#>w<+uj<QW9XV}U3ZG3uFfk%s#{B8qTn9cipe?*`J8=<2o%H0ECsFTCfcy(|79zj2QB7Hz8yiDUh<%HiqWN-f4i39QPr`(LBnPBQzXCcfA
hV050nj*hQyz}fZ;44l?_sRV%lJ`@<J$6Z)($5M07bV+WM$E8XzMZx(t_#wm-<v*jgd<OeCTZ}r6$qUJUj6=ViKj8&uYX#}!sACX_KvD~5Zh(xD&;gmPrKlT
Gp8NieM(t8GHe1_&XL?grAmnC+`jGZ9*sAS*_lIw2JpQ^GOLM1e50)HUOrcjGM==TELJY~E>VB<+Nm5kj1Qh)G0{Tq`E<fpi*E3LbY!1beh*Ik`sr6}=Z}LY
ws(Wa{L!!yY|-*70TyB!lU=v<P{l@YGTSa3`Hts4&Afk!;7i+fQcrBbepT<ct*t6VRj9>8og)#)#dlf!Z9dpd<UY)5Ctyu%uIsqF1GL81iu=5p@Zs3MLj>Qf
;M2x&UNs^P@5)Ywm`C~H)Tgr=V{hdC^Y4mNGKH7z0w5N_7b<%o8a%WP{t~Wx&r{7^I(I$?$*nhbg|Z58`3a3g+Vc`bR_?0bG4;X2%&-HF3Fc79NupP;@J4)=
T*|FlMVOhKpsmR;f$iPD_bGPs@LoB3k@x)|NIT|C`q$g=XTIPk3BNvC#hx7HjkSlxmhPOXus4GEE(bgbHpT68MI0wqN<kwVJ4r2Si(vMl;hSyds0q0+{4rG*
L(cP~oh5dA&#8YJY9@;P9|<X?HZ)N|>@jP*rVdutMbifp(UAS=9W-!`3rany@=<xAm?tpXp%`L<>Iax*Xt{xdX2{j;yb&a7+qvt7Oc78pTmQhx5z}f^u|Ip&
LDOB#YJN-sqWf-;SiRTBU~I?CV7EK0f4Vni)AFN-+BjY_%K<DC@nO|_xqC0S+5HfS4J3^PI|ca-@R<AS2}wCt5G1P5S8Qs)!JL7*aLNk0nys9{+{RD|{V=3r
F9-kgQJE_7;Yf)+Gg^~<3H?6ob;^P&SUVgz`1H6Byj{P0c{K*2MJeEJgbNK`h#Xqs-yebRhPM02eWg*$yK5ayr@hBVIU0Gj47gKSr#bvn5&gH51*HQlL0?Vo
)}X8d9U_Ucog>=tJwNlgLr4!?xsxlOiuLiYdYRHz))vJI=cPY(+TmAu=ZIdGD`Il$lq`uHG5BKmfxLwljJGspcg>wZc;7fnrNtfB=T!%*qHGcR`d^9A3oZ2j
lKS*{ug_^Xi<RBiQ^jDJ%+SLS8-!e)v%?uxto<^)JlSN02pLa<#KT7Tx@bx2`kD(nl1i21CC=zy*_7SzQ%5uJy@x42_E>v*uN@?(pvrUj#@Qh`d^OtKH8K?h
H#e7T{k9&o_(KmW9~6OU)GgAiS7r#k5x>*8rG^svAKkrz26!fEru?K6P&_OfXjNv3jpJ5aUUrV4E^&O3%y0!>4yE<FB$XIU;#(xStO=4~IxR{;C!FVupdICM
1Y>Ca75&wC_&q=CwZ`axP+_0w<E9F*S~}7Z+$e&a%s;_FtCyjYrC%_@uL{S&RV`&FF1%qS_iR$Og!^>gtEd)x_&@#eCi<a1W_VWYGHYz{@4&&nRc=GbpHcEQ
WDCQsemCCxpUkkHmKv~S?Fy1y(-S8ooN%N1MgTjP#(z3hmFi`wxNeUE%VUhO9S$fgKkhQ(uY(SS2Uo7Wu|(Edhf<G&I@Ana+^m_>gkY_zHqkFfhzXzk*;~?y
*?Y8If3NQGg|+srt@}oh`qn(55?6<Rqx$qkwo-5tI=s)tK@BD&o3xLVjiG4}{`RAqDMk*YHGIsohSjgw)1js|AU!199>C}a=>ZOV%Xi%P?x&-{)M$$(bsd>k
USY6%(qXKo7Y&9cwisq(S6Bq_%lm!PgQ-oZ(du*%4tVutlo+_-Ugd<gzEK@+UgumEjIf5K?6DniBqCsDnR>S;1?J|x<Pi>m;5|$Kl9=TRE=GR8+I>?U!Mq0*
@+Sijo%-Z{%(^iakB6@Y-m^e|^}e#^zHBHjxi}qDNJ943c=u@MB<Q^x(;^Z=;G3ls_j)NF?OrsNm5=PPeT+=`hOZqmYA)!kO=?0ZUf5>+vM%^9Ii_1ynB(SE
wxf^fQ{eMP?0w~zJ|zC|^Pepb##|`}F*lPAI1a|VAgcF4HKUijLt#2P!$WTGkBmTuY;B!gUmcDs-P+)34ua<BZVr!92nu{%|LVtN!Tyr6r{;tYdN=|z`xGzW
mRgfE`*b|^5wxmIhQ}iJq(1q#q4S8p%Xp7u*%kwZ`N@jzPEhNLOBm=kLgW5R@e1NiC^1nKbc;y9PvX!=!Ky8Ia<%G=LGX1<=X?vZUJHbq57U=(_Y;s5+DP5%
mw>on!d;R{N2C+JKKS-$7+%(^Z!X<41|M78cY~UXxMNQDm_ObY@(LRHT>7!l;c&3y3-pH4l-KFVFS)ohP1`|7XNXA=J>xtrCop~xl&)Pb1P6CRQ9xNK0!>0K
Z_4@MeUnJ|)6oXdQ!!dT(|1IxUddzSElZ>j1vuRov_;6~phBl-FxnN_Ua1G|^^gYpzQ#X32#MuLsQ2r^E@Qg!=L0^_YEvf`GI57_bjQ#~zYsjQxnIgH!XFO2
l}Y~e9;lvbcRqJ66yii<uQyD`;ZgChX)`Yg8l$>J<CFtv?{z+;UEK$<neQx2SsA!+nIPb|VgTBPRk~L6jA6gR>?ac#35ROgtFe7GI8eG0Q77pLrPPr(Mm;{T
=n7eW;-m?h`y`RA>z*JUj`aSxu7^6so$Jpvd2y2K5nFazFgkBUvHvmfMLse0k)recP@8*Qzd4r&Ntg3p?jF(Llz*bwOYZ|?(Z&0(15D9OTjwy5=7`eM&7XU#
bwO40@{iY9E!3*GMnAZwiO>C9{l;soNK<?sck#Rs8h+lYo*%bFoOaYn?jv{c)%A5#`vWsP-xk@*>D5Gj-Xj~1W+`a>itx4F2!~YBej+Uy3nbTAq;<puVk4|%
sc<qK3{u6ClRK6OkPhJM3bDdA@8jPSv!U>9aJb@i&kmE`4lk_7>v1;h$i!H)K8R=*Ym!Pm5$}A|#p#I~3Kk<WYW2@UB!6yD*IpZ(tdp0P^MQ$o&^r#H=Wy=$
nPWS5brDD?VZNZCh^kiGO8U~h`Te~^anmvZ^!y>ik0b){F^asCx>6MC&b6Vh9JFwvo`uGpR2#nwM~r+|D{$rae!2CN?r=UQK=X^12WIzo{C%GUfoS!@zO%#L
5b5AIlyOkSX8Yw$OWy<p^UrQrZ3;v6m!{*>NIkp>-MJ<}9Dre}iVMe`d|;rz{Zg!;45fJxB301B%6{4;;aOR{(w+;d8n?y4pz)BG-ecgcCn@&a+wZ{ZZkA#X
R6&`2L`T6^3o<(oX-}#QAxY3pFr?HGi`IPUr{XnHf8XfoN^cZ4lgr1RLkju7Pk1N$dB9la$r!tU9#k!UKDRF7_|Lyf&fH{YYSKnx?$ld{d-~`d=rJ!`v_`@k
9=<~Ve4G%8=p%d?1y)AaH*#TqkPM}EN_}dK<gxT$g`Jl%`jc(oboX9v_@95R*2s@*1m+(vu>0e*kpJtKO1?No@mz(^#RgxJ=FhnvNk!E9t0Nv_;W#uu6{RZX
0pYl2MuiXY7`Tul?pYWO#<dLnNqQ5EWFPlcsmVq&b>Q3^$(wkn5^=KnrZbj9Ebs1cx}x^h(bcuH&Y-G$OqL=M51qT2_f*+Zv7E3atep~z%0H9en!EjQ@-HKi
(}E1<oIKo5l_bHq&5f=vI|7Q*i(6CmaR?X<EV!D|h<oebwdAD2Vf8H7Z_YLuaSQVI>Kr1`Kcwhkke3R-bK9aBZmvj3G$ocU*1=Do`KS|Set5TJ^<sM|8B-<p
#abzu5W3y?PQWW13Wx5@EI<dBa#mMH4n(1rBY5oqTPSS(3&JU?qv57=Rqk-KF0KT=e&!n=50f$91C?!QI7Hn?+g0KZHLDxLpUsjHLRmS<u$u{m)S@;L8AW(;
=IT)0c119eu_Wt`6I|1$vwkz$B6fY*t}9a&U)y~=Wou$VXI_}caV#4Ic{RWED9te+VUb{YD+wZ9m(rO+uY<zs8Qt5QC=gE^qC9==4oci==O127fv|EqU%P4o
9B4w%m-Hs#!$9-!iElpmqp-6sA5sYRxckyq)<aR>_^Gx$L=Ue$a_r`kWA=KE>eUNVw{Ry+>QKmtKXy1eFIJqofq)P6!d)7vu<$Sr5^t`7#Gk+98=A?ms<=tT
bIc#Y7d2Mr35#IlFgp5tZ$2FR20xUK#6v~!e6!1rE^aXztB*HC;(|HRbvaQ#kjY%Tc5F8s`iaUvKeE^3bi~tVR78=OOPCTJeh~s;5^-UXvlZw%=$sk0<qd`l
!sVuWyze%7%c|KZ8n4f}2VIqng90r{4IfK1bUJG<yN(3l5XX?%>&Qr~d_77?cq;-7$L#CB5k!IQ6`j(B6Zv>CsoO*n>;?AUy6$=WF^KX`3@6@pMoj<tx2B%C
c*4Ds^i9+UKBjyUT+e)Q!rrv=`a%M(kR2nISm?mAA?NL=p&HZ$XJt`kCcwOvTr$cv5|q?W4sjic#kCAtqoz`K_*fR-I>Hl<?P-O;_w=Fv>2#B>s+hswXV0(f
-v8yyK?@z->WpF{NLu^GEq=7&EWJ4IvWf#<P}TF^A1%P^hi`?Gg=}ERV8Qm2Iux8=d4EfMN(I4Bz9YSFYw>!Ah0x$oB$O`)O%R-j#Qi01lF)ohq+gyQfBYf=
Px+=FiMtj<_DD=h6qzTEw+;sL3#4L3Jn&fK&m_ogw)RZVD}lngjqpva55C>VNep5NhgMNtlSy(S97mWKNC~V_wBn*J8|{y)SHIO0>1slTn)BN)e_PyI(#!P7
H-U8Y^Z}34rFbb~EkTv+jr6owGIfu1>`&ZoO)arOJo$dlK%)Ru&*z_D{9}n52X!kqXk1axo%rY7h9|no9x#TVw8C*~b0U`KE@&VaKbKS}3pzflt6D-@h@4!B
{HpAT(=Ar(ib>W8y4G6|?hy~pxKqBRi_YNj5F61l4T7P7rp3Y;FQmN5%aT3mfT5(FJ9kco!a?EwOwW)BG$fOS9Dn)ZT3g&}50OrMZ2T#HBOwL`63bmFUWpi@
U1}HJO2wzy%8qJVOL#|`C8)&uVRfjIr_D4L3sclo!A3SvaMf*4|KSVb+uE}}9a89%y3dz8YK!;Ro3DN$3d6+-2Jyau^C0t2kQ>Vj0E3BOedrf$q>YMD&cx)y
j?8h}#L5P9tbJ+QOt$FMDF00TQybSk*FQ*>+raW_zOt&KBC^YK*nZo)qJQ$@c*5IAL>mgE81(pq(ru`(W*`^J+XwHms)RyJ-|mQLqcm1!e&6XOwZT5SZu4gP
U<gM|Ub=WU2CwVNYXUXBVSD4d`E@2woa7_j(v6HqK6a>YU$93+wpH?mi6ugkS3+McdxLa&!Xkmi72)$8tCnwlknvb%zO5w~VQy-6{|sFA{QKLpKSF`fw-NOw
6SjajMSfnYWEog{Z+P;>sN*hWdS@Mr4Td|@AKvA(2O&+@*dM0|@V_<t{`8O*%BK1nFJ34>41I}L;T11*iU=R4*|i6yV93)&vuOOdDbScAW{Q_*9+pqhyFgrS
>{szwThxqbIDVB5f!M{}+P9V75HJhOF<glQXGnB4pQjnJTjf(}!ZM(%>u@TGClWuDm=`WQ(7_Uc=hrRw7~Js?v$%5B3}Q0!%Q1fpaHh9n@Njnkrns&@%e|Zf
l1o4B$=nT*n?@e?cTdM&)uzK|-}!-Jf8rcvLm)EMer$eRcE#mHhK%2yp7^LYrPAN%gN@i5^V|QlKp4eU=+c^q_v}{|Zi$9~-o5AbO3ihQ6n;{wBTj*eBYo@7
uqf=#`mY_f4@7Cky#aPjcW4jkIkO8XA(@$-y@x#tEf@4Rh(&xc)XP$;|G^)%P0yW0qI4nE9T@#VIvpiATEQ%tkC&5Iip4FXAjeLS<Zv|#H^tdfVx%20VDv`g
X@>xYQ)y(zf5)TZ0woW;0`b!+x}Pag4BIoqD<>m;VHKuEevQ`zcfO)8CaMxA+8pM+V=~ZIepAQcvIXuvLL==S2T;77r&;^CH~06{zpr%oVZqIJboXclh*xJ*
?w4gl<NdX~FIT<s&hhi%2TsOVpOpX0>5+;`VjE(j;x=pvrk{292!N-$XouBxUp!u|izyKZ#%lX~-O!Cx6lHehlZ0l%t8(_yyk`KU`@7zY75c)Tn1if4-Wqu+
$>VR+RIr5i#~Y4YK+wT?F_ggz4K&K3w5Cb$vdw&W`oSKDQ!x{~-wH>pLv6~@({AvJl-=@t<BAW_9o=(d8CWCL8D=<Fh2J!CM5Bc9INc#La*f9l)bvy1E$5x#
KQk)5C=iN<{O<FQp625C(_RVEk9`06cNf=p>2uB3(P>{I*^m|r(;vzWd@+7Vxaa9RFKdOg4b5Of4|_CT8)qW@8VRHK6bJXe^n_&Ezk%K{Bd|shMRtn&;L~77
UP62mUikVLr@o6ryb5Kj^_K$tzP>B(Ly-;VX`z3HdmPN6N-L%B;tkQ+j8VGLJe(`Pck4TI1e6EXvrmmjAWF9L0~vQ7PNiODq}+_b673xCem*xWWxnn{bR--N
r+V&W-Ydsaz?Bnzde&&(B%D}%<AD2VpTar}f<SfYzD8u#RjBdgy>sMOKuJ?)@js3T&~yt;=YEz!JK;mr?C}`!{+dAac!PDckDWTh7iwFpEj5+;NH)1Z&ivON
jL=;@dDb2zI*DZIayA(1+c<uEZ{Ik_KMoA|$RY6atq&i%D`0eTL?PvwCv<|!ZixSh!t{$V{oCP*xYIrz_+Uo`+dNG#>wbG7+{iLIDb5BR!-qwei(PS8?{-2$
`31NnSrs4c^uoN1t@D(eCR(iw%Z80yATno_*5>DnxT!DP&tE3tTm<izqXz{bu;2Mtgnb;0{vFtGd1Q(VkzSI45<AFiul~@Emw>i~G2Q-IPn_<hv3~6C50AsC
%HL=#VIOGtkYLOM^h@7KPYe5EO7#)#U7iHEWR{oev3kPO?)KMZet(F&yf<2riGm(OIJ6xrV758n>wCZ(ky@K1%?(kwbBL`|V#NUMzDLjST+xT=bhXX`l{?CB
rz-O^`(f;30=1}800tg@U15(kg7(b=+DEBEF+5)p*|v9nBbT;l!Zx$9!FkC|V4pQeUK%PC><`1UjfZ@%mmT3VZnysDQ95FYLi9~(WHGT6&*A;X7J3gGPJApd
!bY%yntzQg{wNZ$K65t0#GBlamqV5~89!w3(c=mZR#?_Fbvxqb9d>5!2vZc4-&b^x^~HdDvPMg&H^zBHUVU!O!prIL2d*>@SQkt_k^jXKa?EB<mpWZQC^Hsa
a3lz-ItLe$O5KpNWRtKsV-J1CQ6t(0OLRM?PrgvGMknV_?l*dxNX{GczRK-~W75_WhibIpfPVS6aq@VOr)0SO)d%llY<|$#Mxl+isN3RH9I9Q_2^owNQSq*Z
bnbLL1m<+=Pw<A}+u=pC#bRraw@Y?MDcIwVX;D?X`v2b>2HrdO?XRFM=3^IbskDS3@ri_8(Q^kpUw_*e661q#rPK86Q_eWd;nSwI?GCpJ5v_ZCd%cFGY`Qkp
2WLE|^u2hU@h^wL=1*cF0^)ky?K3^GKCb>_QqT&rotYV8y~e2eYv^EP=!0A4^Xu;=gAu|u-Y2bP4;mK=S*cl9gk4}DDA2RVHHyORr{9dRsOYuHv6%{$7ef2R
q;la+GGj8`+JK70mivCE>)_wSpl`*Q0l##+krvqu<bQg{*(l?Sb>B3novZ+yzOzXb?U;jQ(OF#`-x}QPeLW$2<t}d0F{kZlWx$hW@wd}b7OKnsMV#dI0m+Ve
f6VP1)SM=iD-wtSiRQMKBWn&e-aWnBd^Q)t)32l@d0J8QWmGnQPxo}k4|Ya<Pk?pM61TixI~r$5Mgray!o=nBqrmt`yt)$Mw|q7N*Q64%rPKUDyBXD{oShB_
^X5hN=t?Y~Y@a1^2tjLdYO#%IHkznMZurf_K!72%Rw6tcyY@kgX`Pkm|58LCBvJ{6yJadJIT5fwUTb;UpcdZ)WO7)WQ(=8xk(B356<FG=uC(RGLgCF$!v~2p
d_6s%+tSvKmdvX(bA>e!{db1fGusNO0y`wb_Gw@|lVjcF76^i+p(`O`0mx)Z?KnG^iJXl1^2;_^Fk5(=$ai_q_v(tL9RKBDW%1&BCZ{B*9iEFWSt~-4HE}n~
?Rp61Y8Z6e#35Yc4bkoEjaX^t`e;!U5104{JR+|%A$geU<MltuaCTeQu<0v<e--P6iM~u^klC}Pz8{3tc_tCk_i-S65OnA-VJeDWdkwYmmq7bF6XoMKQP@5^
QP;y>jE~_BYw0&*At%BW%|9Cn?y0pa(G^A5C?)bEj1I*O+ga;(m#=}`cK`W|t{RXnG~YOAQ3Bzran_srTrfiD*6_SN2y2Pey6ME(F#fB;$*gh>3<BpLOy9CY
^UN*Uy1hAy@_#sacdZZ<(~tU8_IiSwW3NSz|K9l_m!Ut99*TdIl}WvVwGd-^Qg$~n2mc%j2uaUm!}84ju$;PBP!Sx{J2f)|<@hs1L7JhU)8B0$CrN?yv7B|a
q)Mz><ys^XCBvTe50|f8HlEEsm|C~bLGt$$hwq_*aOvW+sGrOOaSeeX>$@DJ%_Tp{;*CVsihb#IzcP3y9SW4Z)Qi_Ceci@XAz%zV&~QN66Hbj=WWfeHu%4yk
P@eNa9JYjZ@?F8?f2EsO+z;m@XNYWv6j0o|%xT!Ef%W}N%8{S_!J2Jp-ZP#8nI=vnqf^<a_9;k__~wW2FO^z!%;WKzm%Mbws}O7g$38|MN<*7W8BNS;7EaSm
Oh4Cg19=aHSFMLHm~Yt@>_4>Eue_u><o{ZN%9c8c#kLv|Yh+UVZdWjJdYaWX*9B`fA-Sm^L(oHX`?h#nIXY=n+O3K6@#*H|eS<=G7$`U|hw2*Q+hBHgn6oxK
g2PS;2YMpm`h$hbo)%CPIJwoXqKzlc$L-xu-NDa?2V!K-)*x$Kvg2cyIf}mJZ>xHSL+{|-n=H4r@MYqBX=L|xd_BiZzJDVT3_g{WEZ&!~b&#Ah;co$KNt0$B
gUjJnzD4x(q9azCf`8vX+6cK;CGy!}eGqm}t!^*v={bSBIQnKMia05CM@=$d?)BEB)Y}`*imavT74Zm*sZSLUHbHsJji(Bfp?L6t^#MVA2)<~%krcS#1>cK3
KMv&>qK-9F+O9niBIPv$FBNhi<Tf4rxhe?-CYApV(|F_VM3usiV>x&(fAH{Pu_?;Dhz~O+x8iJXnSWhJAat%0<oI)1{{Md+G;^1`>OG+mnm#9)nueqVr3B{D
c)XxE&qO2>kM;P%lWDGlxK|oOv8Gjs>qGvLyNapMWbW!zRB(W_$-5ExfkbpV4(zjXlY&XG0iS5KD}Fv<;6jQuL?n-x^K_OVdL-pXk9a6}Lu|PjXswVbe}6EU
$R0gUzC8Re7mR0QdS8llT%f$)`F^K<3gl^atsejKfdB3W0n29%NOS9TG*Y;s%#p%2jwcH0WugWKnX<^0lDK~Kq$NV9p7x|Z(MNmz+nD~eC=h<?vg#jTg{3p!
<7cexsNQ+EkPt_^$J3)OWuE4cE(?_zTZ(|R#1_T-qP_g=(~mTJ<O++L<I;9II;hoc^3T@S!-s3320itLC=-5U^jJR>Qcs6&Rhda5Y9c(ib+2C(-E8QIP|`&)
i}=&HAT5}@mRw_})Q0TS8~ffr12iT5lg2F!K&_VWNC0N|tk?gA+*=!utH?w)?SVQ@JM(vPdpYZ^_I>S_6#QGA2da`RF}8bVn_SQv(njKB^khC5FQXewXE=pM
`>M95Qz^)3pQK5r&_dDt?Hk{Ht#Fd_{QK7KiwNA3yY5tPjAbbvpSXjT*o?HjDi)^)azDy{0sN|96&<ZS?qP}(-w%b;NE+a|Qv0q8ttBc6o(_K<c0~Ho1atMH
mbhEAqw8WE0NFp%k29?#ammk|kch}18hSt7+nSs)m(;h-M<0QHFS%^LFcX}*N&Up2TMeXd@+5|JJkd32yI=R7BgFe}j5_VjMWCPUw!VlZUQCeG%|`jae2WsR
<;RhDF7WR+3OyXARPE$16-C#lXZ4i_U6DTZ!<s;h7hfyYUpC#-L(AM52Qn=Yyry`Oa&T|G{i3b{8%DB_@>jngVXKQd15(MG!>0Juxjt2%>xzMs7Z0VJyMV!Q
(qqP!R#2m7%e_Lc4OT9CanqeBg#TWRJ`z;`XM@G78iz0M<(`{5J-r`FojZ=CN{2((D=gjVLpT;lCJIGF6|wGA_EKKN7<W#;ys$4{8X*Zc%xZ3HLh)bp2P<NE
=rMIl2gPvW9Cd=&tDovPF-w;}FdcxS0q>To_xhK$)Pkn9nJ=Ql8`xdUY;h-Y?m^X&R0wx$-(F(UMvdgBH3Molh>!Jc+_QB-+XF9Qo>|rZbh_Y1Lv-`9D;!1Q
^cpFB5K17ub?K}zwnwTX<pWf3{F#s4(?LCaa;AU3ov(#g7Ls3On{45i{c}d@S`nOh7ZPl~h2oAt?X|aBwosj4I36qDiQjI<`CWa0Q}qC)?*S8Be)2wjgwqN&
A~RFvFhnh_&Wb>p4jLC9Kc!jB0M%Hrs;#Ccx^4enw|S|FTGINkMZ0hiNe)Y=H+Z7uI<=R@VM~-y#0}(q)J2P>Y?|gvZ;a;OV@?-#MEn40@a57h%<_>)ZTe=T
ud>T}B2p8+qiV_4f0NPmAucd)&+oiU6~wteo8Sc4&EqS4L228~Ij*4tb?Y0%8S)umpK7+-2{gmOPXYP-o!XG<vgpdW=7k6{p5WYud~DwTKcdb%o~y8b;~G*z
MzYB&yKGsPz4zXG@4fe)*;|N=qC`ug$f#5zLXr|vk!UDM{l3p1zvubu{B`bg?sKls^?6_Cb-(UoPI<7h4oqF%4z(9uFe9o{A5j$x<&+;8PwqH?js7Qn)utH?
FB@rBcN*Z8f)4NeNi)no`f&10tp^<VrIlZHMx#IXnUZfw3689vpOF&s!iA*OE#j-n@Yug@|C7`XpYHe_VAYDk`7cf_-k($8aXTdZQ4^sbUDgnxatj5cE5FrC
{XpDd8r!BzBIIt3vh@9Q_SkdEGQnb)8&XocO>_OBIHUNLdgenUuB4~(zq!nfQ$@*(zM@tra{DIvi`0nlCaQ!F5%O+EF>l#@?ken6nz11>Y6ize)scC|GJK?-
i~cN=3GSa%2BmiAP!O_~JS$xaWz{LyDS}`8lV*Q@|C<p42D!AvMT5bc!kv)2W($Y)3xf)_su1Cgpl$rmdUeZm%FL@_u+tk0duZSe)sGD2kCzkCd9?rOD?+Z`
w<9oqZA}xzmp^>pz7vblg)`**y|K{PBcV_forG6Mr!8{Q+QIiLe2T@*9d4{4?3edCBJG{yi+7*&uyT=n{+37(rtO;B>$jrNAmVAnALfoL7d7{K@a3ZF{Op0b
u3S9t`x7_d=?a71tnbW3uINy@JQI20JjzQR#<S-IfU=oUtb`#LBpl4jh7YPRoH`;gA{dH%H6JDpnQTOsW%|#QwqkhW>x@WN4*u{ar%w>-kBRK7{^mbjyd|+8
_%s`U!f_}?6O|)CI$x!B(H=5P^lDZSK~M`2eAP^DizF`XgI!7vXkJ;+_j55sm9f06m>Vz7^1G*0n`h#az4#Wlg(DOXMlq-A<m2_fnJ-F2!BC6-eMGrE8DD;C
ebrka@GZHOwy?Md+Ube5w!6*ngj`A^#Ks;J^5!4s;0Z3W8JYHJeSB(s(UNwS0oQ);mnaCIhMhIFN2sJ3LPXYRoBp$(SX-RRx`Pf_IwYnloQ%O6UYx7(PzIMj
aayTX6o6&I_4PK38HC&3`qn*(Ko?6#WB12w?5bNn`9W_12a<<3b)J~v5Xr6Sy<Kt0nryu=chDC@gI@GYgq(Cxv6v%WM*_(W6Z;mQTS7PS&+WyfK=9Qo%498b
;)`daI2BCrR3w6(+(Q-)9S!H~^Oc~yzmRb?!5ClaPDYn&O5sYx<khvW32+My;f#hK^gelcm+=Lm`rMquv1@AR5{Vq~v@$~&y^=@%FJ~<BT`Ob~E5caTGy|)g
7WPE>@C80P4Z1#7a#hY?95nvR@=Ded7n8_7eAYC^-XpD>99;nznSM)`wci+;bz0X1Edt?9Z}`D2H54?=<}aF6RABbzx$(~X1gwTMDUuy?1<Rt^g?~4+u&pky
LHFGn+c^dYg<mG%{NIJ0)r;mhYbfuWxnP8shf|dfepJTXy`h@s{v@y;CE+$Tb_aEo3zcH59L`uDD>^Hkh)_GDv5{CO>{H+Sst8W_6SNZ~c}Nc<TgrE{DgO6=
Lt0s_37vJq`A(ck;?GXtvt-WO^f&4#>)7}cXP1kcs(a{)MiL><Vrr|QXoKN{QY$*FE>JAWh~|8(2}eb1KPEyRZwQ=r=MCe=)2N9P6IZQZ_2J=ecHkL=1$*TF
Sy4x*tZ3HOC2f3?efd4XRRh8aWVM>>ZeX_Es1DSL#hXHY)muE7c+D1ig8r8;h^L*#0)hfCAZh%hIOsGMZUnrzHeZ9Pjich$MWwjwtr)FqoQb6%%0!Mm+W2Fx
NJpNegCCA=W9kw%XdUnROKj(Zj`GcfZyFY`Dlz$URhJiaR~+8R{#3!InCD~Hvzsw$6kdAYRvG-{eaaR;CGgWhCBHN$1je)bh>TR@;Yv35T%ptmYDvFbZ2Khe
YcG-gK79f|{-OL=w(1H6=5$skJ5L-|-oK)mD~g75rzV@Tr0{IIpXlA^NX%G%z7>=yit|=pA<CqZ7@JR-cl<02!;>Cn$7gxbC#ygCVpb7!<r3rc@k)?&tYctV
NyTlUhiNA^f#@=&$EIHt;L#%Fd`tZ$I2@i?yScdF&=dagoIgS6<;40Grc6k<602monZw9Q-c~L20wk-HP2cY&jHjU{XK!K$+kmjJ1C4=jqOTFS{mBsdi6o0&
A-dQbyGk}brH+n4QnJk>4!G4`l1Xi@2?n2oX1)0?c&c_$y+K1BHCt7`iqqZD@axf=@qAIRJ+BOiY_bQb&%%m9aWF_%Q_AKCB|+S8debQ}4r-ZNJAK{aC<}a0
^_|iHooP}otOLo2_>+?*bI$=Z-AmlGrW&9dzD~i!bQ;A6bNI-sByc;vO@m)u7gPqJ0xkPuz<O$*Cr^|CYSW*)$q>Nwu&{%%`&1-|nSVU;k(bB4rP|1XI&plY
?a$gg<&K{W87XnhK3MTS&LuD*g53;Cre5MR81mcVt>&k~NV3LDsb4nm*{moPoY4an&t@AvH77DXSf<px%@Nku)vV2HhsnH!D7JAe7`=Ga&q*VQhArFfF=0ov
&Vk8%G6{`m$rEGzRk86?PgbHz7n%tx!?BdEXbrr7yn~Al)Sau*x4VsC{=?|N*8(++WG?Q}ON>V=H`(qBF?Cc@zppN)^F_<KaXFUXnlQhO$mj$>ZY!{ncv%xZ
*5MoSg8$opW#YSv|F01>5R1+IXiqi4@RP=kIdN4KRwXQj9uq)WnoxClZ79}Hiv=Y(YC=r!?Oaj490b3U)$7yggZ_iVmt+e=EIi!OWm?k2$jzm^b#iIMj|(T<
JbD@~qfg}QwdG)#ztOpEc@C1?9oD;v##rdLq~NVPj!@cZ`Rn8%sPTU;X>TEpJQ=ME!W{u<<U6P5MC}LF!AG|iH(c;c)a7)SD-Uv7?POYF<xyJQFeqD63Eq()
s%btW%ybmLD{|HYWuHhOO{D>dYb$cz+*3f2-l3tkdUdqQP)tQttHAt7nwof<0Qjt;&SgJP!?%DwVWa$Ve2Lr8eC|<;KD<09wUdbGy!_t>jKcATCCFJ%&=?Vv
Z8p<>{@9soEDgFUin-lmedk1iac{NN%qw3P`8HdZrf>QnYM4Fdm|Ov-r#smD##Ny<5=g_NkO9S|loju)T14j94)drvBTJ_{&bVI(2R?9__+B!^jRd|g9qchU
@YXzGz2Ow}=;zw&8qHvyC9ze@6oWQe$Er~>0u@nC{BAiAiNsJJ+4YCQgnCfoy=fYPzkj~nkBJsTw0R4$PP_u1xY8e^Zp{VPUAIX}V?P{_f2xrwrUp+EMJvfv
S2+9)P<T+!jJ{)>{MtthAa?fB<>nntkX?!@Ak7SeSo2JD@E1uG37ry7n6-iOy<5Y3rxj4_-%F|ZE(!WfXDxLId1tTvn(elr1$1cwXGxZwQ9R*lvA50_YNc!*
6qg-Q-zVL?U1o-br$(_A%b{@Jy?S0&p%9Oa(?Ube6Z))0SLfQ#Ktwj~9XxdR44zdo6_wP<Aw46UGN;lH-P1mQiY9X~R-F`*)NF=%UUx@#(h9ux>uT9L8;mlF
a>pV{F?d&(JSTw`249`3)F=37*vt+i#iA};zp9<S`S}Ej0;vLy4oKo&u}I6*jxN%f*Vbx6M6h7>ZAxs7(BF;<rqPuYfb`-KJ)SmCq^!8FY?WK!R#x)&M|y^6
39j<4-D?H&j=jC7erf?rALtm$%wd;sr?1%18S74hr82AR=-MxGxT)M6XL<&{L~zI>YuCqSaNQJaw-TD~|4c^xI(fmxesLUW-txOjsJqfBv*fA2fq16g_xs(-
1U!Px54_zb+_&$yM5jn~sFW|WJZ(~jp23UI!&%0l{nuo9FoV!{?yT+~j+X=V!s@8(fm~EP8oHkr$O-4eL?j$9ba72J(NA$D4DK0xf|EZK(B_*J5YkHu(R*|M
%wN-D=e$tjfie+%uznMH`kFMZh8#!~RTstB6CTdf-DP+_7u{=M<N@RNXDDXXRFR_4F*{A>h5QlMbN#fuSo|)+bdFR6?qr9X_GWNHFuY*>qRJWg4y>lPh8RJ6
Bf3>{A_E$qPG1cFXbZ>7h9~jn7*xILq|dU5U~Z34O!Q(n$Rb8Q2y+~TPn1)d_p5A#NYv6X`_LmHNFvqjn>@7Xm4-LI@PP5u1scAk<9M>h?=TZD0lHkQ3*A-0
rqs6wj(rx`30>j8>a7gBwE7tVVkcOAdL2R?OL)#~d#XKrjA8!P_0go#8Iab_jmX{qDo@hzZ6sT#6nTe_xhEeoM{4e9#@xV3I9*R;q4g@k2Z8dx6niu=Y!>8u
K*$NtEL?WSt91d@$70Ur0`N=JU5?prkKRwMb@~yGU|Ev-v;NK)XGkUJE7NRnr_Jzo<X0YyS`9tEcU~X(_3D1UoG)Jfk?h*{#SX;3(_aT3l7u?L=bhFbEvS0R
8KRsY&Q4+>D(ptUAG!9A<$~xmReD34Cy7!vpQ_n|QuvxreIOxD_`m*b<T6bvCSm}q&XqPAei=NUPy89T$%kE?ccwxe8b~SK7KoG1!YQL4r!`+mWA4Z+TaK(!
EN$@oBfnFM=X;``^uKh*I40|u?_9u%UElmeq~U1c+jnyCniKL)ABiGbszz1JrTXG>2aw&qTh%9`f(AOdmUp7|U{$!!tRLkHf3bUY5r3m#xUR6QW1WUy5&U}v
I~+lL!^-iPqck#0f_`2U_5w?U&<yWdFm`7r`$X!rFe^drCB15fK)(|S6m%vqvy%CB+$j*@@>X_!VLYhhHS!V=Pl83rkyjSy$WY=cwNBY#4AmRGF`pCt;H>$W
MA=FY!2?c3F5NB!y_||Eo-@baj*}+*Ykn{_y2Pndt%}Z%X~)D9TrjaG@21B|VeDzzBF|HG0I~I&17*Df;>|YIt8Qz<ecxj{buw8*?<=(WTB8aLOSEKm`oKkM
s`~CrO{BBfdt_AUU`eGr{nsmg+~N!|ZD;j?a`mYYu1+DWystF24-LaNK03M7IyXF)+MtQoGr^!(VvwT|8@>#<DtW(+L>cus{qt}`n7WEy75MImVfBar-F`Y?
_NecMmnnFYMaduNsNiF+Je5VUI*ztX$UQc<N5pI&Z`?OSj0K|kY?c+S%Lq2c-POkJV#WvhR+gZTc&lV!?~X^&T%S*w`a<ySvf>vxVeD>q><&7Iqr#9uwc@@y
nwvAMKimnzeb@dD{`bakt$8!tqb~!uN3X>wE?c2={@*vxEjehu)RW0>P=cbGp?0UXBbLgp#EJcI#5^shUgrEsFuv&9mUUCdtH%@E4uyi4a3Jg5lypFi+1}9L
J<K>Z_j9Od!2mc$V!_5?fVj%S%E32Y;ExkOAa+_2XAXo!@6}TR!_Tk3TuY4bk#aawI?DpTj&n)cuW>`Xo??Vl+X#+{9}g=X(ZfsH*WYsI9npWM(4cq61Hp?D
k0Vuup{zF*AUcx_DtbqED?KOVw5wQIyEx-KWv*7P<{8w#l~GC<)P#Cu81u_BQut)S#Ln=^8%O89pY4`WLqbZtXD)*PJpOiAZJZZIN%Xg)T7AZ-xWjahUs4qg
QfYy)^d7jfRFr0Z-VF9F`rUDE5%@7P8aJt{gOl&pvsj3|U>RmqJj88{Ltl*8E);70w@%wv&bra1dLdqq`<i{M1!9PdW9jedz|@5}EOkj4a&Bo)1=<V{Ka<UL
-b@8lnVus#-}P~?@b#4(1~=&Nz0jwi@&=E7-d&1i1GvgO$UhWnh`W*Mx1ze)VSaLP-Se3Zh>A@<^#5?e(-Us~HRsLI{g;QET2vQR`a^{*Ibm29{jHv>;)}0=
SB|azlY|R(ZS$d9T{ua!n9lGz1Sy}A<oOrOG5ht)nwFLtf;3_Up8oYlcTJG~AfexDJ*+<{JQoMUuDrjaW2Ly-$Aj@aHSFAHPbO~l!3(Q4q3R}ol!=X*)3zu<
vZYc<I3o-<ZWkZ)3b6oB|5+#9$ZSyVT)!XC8V+5Hk>J=^dql<;&@`pRV}~z=pMuI7Q)CXm_<8ccu^QE~N$mqB!>_4p(E%`YyEhzw1o%z3g}=yj04Z@|?YxpQ
9#C`_`EBc9QIpm&{DVE_`&xBV2f|_XfPGUbs}P&JCg<r{!=TA&d_0f9pDuNL?16E<VD+<ajQQ<_)9p3s1;Y6l6&d{%s8@?ejc;@)eG?E`^71)nktt5TSxZbc
Cd^;MJKciY5spM+0WoBPNa4HsN9R@mR7Fd7D?1Z$?IV>WsnjVfTAw*iN#leAAUU||qKHUgb{nx%*0{TGcaL)ZIk=vlJ{VqJ3!yU{5e0i<aBEoAS}-RAS2Wum
a1SNITe7L%g{Tf>+5$`zl@Va!?P9S0q7AEML%HuuVPIf7={#Iz4%tNAvFn+@?(OKJGfEDy<zDFR{1}I)_r&)9X!F1d)33WR%n@iau{!*2-2&qWG_;jn)o>s{
bh~L!0NM|`{vM!C!hQ$G<#e+Y>=RbinY~?z*DPr(a$^pd^o=BEA0o`}I^C+FjXLa>6ZWf+x`9r+VtK?r1fkWAYIp8>pun$S*ssSQyUEg<vvC<%^}0)J|I`de
m!rl`Qd)wh{8ux@t73FKph%HucZS!#t{0AXqHsZ0Eyp@L4`aWRUFJW=6F5>@r*pap&93FEisf-Q8~%1xm(>^Rn{E|!QX2T}d}N;cmn=peuzZUjcYx~Fy($gP
L~wDnh;t5lfoF!ErcFNxLZX|+8t$q%Q82^K@zE4MY~35u`koL~Yx>}2Er@NV;lo$kEirwV`?`9C85CD38jzojrcn{`zAkUXuQ4l<J<3Am=0?WzbQip?wR&3c
G8iL&`-N`^E1}Tq<J|H;eHbNN7rr7Lj33MnS^FZaF#CR&s8{(YBD6XbM2-j{=~3g+qAe}BbXK0ylS{+e#Toi>Ya_S|>$JY2vj-DMX;(xfL3%by!TE?e%AN@S
w%E~v&!s)|v#AE?QK;}qq>V*f^`TvU6Ae`5b<R1s+rq9|_C;5(1$ZrWm=7Iug5GMSOS@7LPN)4#am(Pr=J_UxB6bs~@5ryFNkyP;zTjQpeJ*_Y_+e_5*%A7$
?TlGV1o8KYWZ16(S=i0X$MgiqAnvhLuGA|rEQTxG3(57znUvr6C7Zp_Ui;PXK3fQuWt5i|OvE56|7q24To>p2wTQ<a8lWv9PRxKk3`3D$?KST!;h)MOm!D)3
5W8}%QT1jP>J0ba<s<lm>Z{X^f|H79`bs({XGzeNBjfN2wIu8}7h0<x8N!O>U681JC|2#de3Gw;LC%Z()G>)HY+Ejook{b-&}q|4a&!v#8Kr;Hroa+bWW7fe
6@9_Mz~7{F+5kV&8=cY*N`SGAaq;&vU6hWGiSXL`B1>|AlsCCKMo4u?bdQLkXV>swzPbnIMI^svafqW`aJk-X)*MN9{#@)VQo~EJW4_D=wEy+*!LP~QvR$E2
YE!kpA!&~@vETB(m&rhR(WZ-q;T+;AkA7{w5RBU=WW%bGjj{Xv508|n8-69T{F;4ff_+5yi)%8wz<;7~rnBz|RL?{!tgqT(Du8zOKkGWCTSyp_4+es<SV+7w
C>)L7JyqM~&C$_)+cneC31rsGESeX+AUL^8z4lB8dr7`m=Ga8y;=Nk`bx|jTPEI{_PRRhX>)ZI+0y8{#etM3o$R0{OE)3P4N+>$=`XQB*8zLGV%vHmDaf4o)
GA-H-6c<asGl(jqQ%LadUUN-oY-ev}uM30D;m7ftBEGoVDUs6h#~3k_{zi^vaTpz;-jSNHLiZqBPu^>D>_23m7qnUk)r|G8S4?bid)a7TZBronw{93P6)K>|
C{C#Cy(4~{ia$XX8in_lwUcW_k_bHcY~lGC2ZY?C9c-`R#{uWI7e>#DU_nL9@5@(+g>Zud1#+=)jUY8oy{ir<1IBVWo&a10-7T3WQ<xpKP8H9vL`075Yd_II
JhClpxWn#-=K}A48ouKIG1-TU*C$=^>7LX+c2@%kz&pc}R~05M`A1j`S@D^yY3M<qDYT?)sT6m^(DvqRk5sT46o(Y{o;0!rSJIV_D_^Z}gV*@7HeA4cGw?v<
Lw8&xwb^(>nS@hRYx2EJ{#asiq0A)q21n!gz)x~(@Ln8G{Ce92nF1n1FU?#Kk=5XO`IIWMinB9<lI?Nq%7fh_K^aJVzqUHRqk%u^ie)$JtkCC?8>drMjO)@|
NjLslBg{6YYqh}#OA~GsI{SDa&Kt?T#TJW|z+A-y;{aG>AL~y38U~3^56kL}L~+8YWAm221#TwW-tDwjgt79?=brB7Q2Uk1v}<OLJ&B$EA)kF9Wp(CQjbI2K
)r=^Pm)hWX{~CxyyuqiJ6M3266GH7zkF7NXLt59<#@NyoyXpg*a;g?M`pWER0D(_v+Zh8l^<0qQv1h!`-vf0*lI%qL!oi7WFMSNH(G(F&!CY#JgDX0;OA89f
_&p}=z-$PeS099$>0PmqQNeeINdsRN#mra=4Dn?1u5vx82Tt4=@gP}<gGvUq;Uv2c#(Y{+qjo(&wx5?}tJ4GfbXXJrFsC7sA(tpT+ZJbjFx}R=>xGesLBW+!
%m3CXsm0wwBX<kXeX7_w&g}#9-|y}i9e2db7~%AN>q)3PCln7UT@<z7ywkN3f>PqWpP6rHBT2k`q=wE70VkwO%svJ~=H3g}EsqG$Z9HRPpteUIyIaboA1VlA
VbgrMqmGX+3xvPPM51|C?p~0bJG!}Mzeid(gUbJ(AWfbXTrwY2X?Z(fcWPf4=N1nnJuN>L{wju7Y-%@~UkHBh1hKyNPQy9=NgkPYFFdLnn<i=YhTfk2q{P#f
Xx^}u7D#iz18Kv^g<EzwEHPT5T<MMD1vdAN1%zQ|*r0EYC=xSvIvY*n0m!#;I>^-%2!{=>^h5&ZcKJM%yk&0&&Olqoh(aUOvD^OR40Ql6b@HIt0e@WTPLF3i
?t^32Hx*^tW1(6lm``!CABC@`mA;*`$7d^_t7dOQ;Gd9n+53hQ1euJT{t$GO#-1qcLFtdnO>dJ~xh!z#*sF<Gr6I`oZ7$;px5q8>)^5A47$lm%$y6zHMOwCB
msU^^w6~6U3lat5)Hk}ZN2h%7x}o}VAwh2!tv3d>b;9vp)UWHXaRRF2n!om)4S|)w>glwdAjn>!^(Ijfz`dX6{8;WdB2G2UV64#-<;t_SwIz5kIOXt-(ajE0
sh95^d~Spod$aSG(j4I?nCZt?;Em*SMN|#HeL&9ocU0+(C!FktSxZjFqsMlE!dlTBy*mpOeyz^%8Glo<wi1LahN7`?UK>mgs1>t&c_7o3<Nj)t5pFAp56>Ga
BG9huwvDzWPHmSrOOAP<W^rP<=Y}hMJQ=Lg&NzYHkA{Cs+Z>MhH9t1Qa*&&De<ZRe1#P+)IH)Ra!RA7Sv23k3&gb$u1#1T(b+P&BOo<5=vL`IG_j};=PtSN4
sX&MoSc%t=yCOZ5HT4s9Aar!^e-?Qbj-Zz_O_mdG_>e|=?4*w`YTm07tB(8Oj!*|z$EkQ+_#3vVHfjjV=bf$--JY;=(K@atPw>g|EAxC>t}s}URR|sl!y(JB
WijLq_@()%T<w|%3ZFT$mAA&=u<NMl16x0YF@|?r9Wn&pjZClf%1H1zT2j*r1mRdwao5N@2dvtR^-&djfcNe6zIVok=rB-sq|fkxuOewce1<Ogvqa+B9*Dv5
XHeW^hZkHP&EL7GmI$(7Qq34iSG>BR)W?o=94@?alg!Z@vb}Usr4s~h+aIMLE2@Qpr)-BK8jWx$<fpt*voXH)4_*8>FN#0;ib@i}CYZf_@S=y13KaO5>RTlw
ATxDaTem9(agv%z=du&9KV2r}omvp)7l=DlsZ7Cd%69v~kSwlOc9m5$CE`;a9i<wREi|<nJ*`%gQOa{F{-umDn!hZm>^o?SL6r&ygEmK)70VppxETaEkBuIx
F@e9K^^=*S-k834lx)0L5T9e?Yc7O@fj&sh^Z5xojP3gvRs2pAjoXaZW4w%UUNMxM<AgS3Z8Fu}I+W1Bm&clI;EL0aZ!9K;Yr?eb@cqb5S4?(hyKGH+11-Zh
TM2#_vh}DhGs_VN2Id%eZPbuvYqZbmogv-|39vMzxZ#9CUya%c4=_Vb$sv{lyjLu#hlD+G>vC-d+i)ms1Rm7i_HxC*ZVoq{R4~fTbll}kEpXrf>03WZQ>?z3
HC4~?#@-vZk3>{xfw643M6WymYGsTS6>&Nky!AZeoTDauQ-=6d2#M>Gw8kH$iviFYd2)R1jvs8CC!<}<)$mm4c17@@FEZ!c^WWR~;gaChZksQ@=*fM0L-K+f
l3j)S_cGeTxoL9G1-)?mZtIb`>g5RH>FQ}+(mL#6*|XLl_rLF1mxVrR?`v?y-o$~0<q#+IE-VU&76-#{Mv!Loi35lqW%V6%afMaw^eTNo9>hw&Nc6awp>4jE
YREzzeYeRSl^x13&FVOH`H=;p^3=kDNL?U0ab}0;atui5(iug&gOMSYGGb}(h(cDz*w0b~zj%DG)ydEs!D||S<BVO9Js3N`9-IK5qwa5N`y$Z%Vd>9tYEQgb
4%=WqYmAW3=S~#u55P69poY}X;TR4P{xnU{S>|b4`jk^vXg7K3#IPq2$wCV)Q6XhGvhw`$o?Q)CL^9EzjyHkf7r{%<-YY=_;$NPXIpNhu+RUBmU`*`%<SS+-
`24?fslqQc!TIT^cf%nU+#7x<-mz?p%&;~p-wU>o;Z*TBNANB3JBMbepYmdJ_WY%%L{1Rzj#VRb$$~kP{_Y9B01zw5wiy`Jple3xio%>GwzpP&r#)TZdobs{
5^EstJdAPcNO6JzOHy^nM^U8x+anpIZV!2GZuJgwRglf6M41;jz}3LN+3=_r9LgB=yqAJ-)4Q30lI#TjrIm&3Y`S4)zTi8fmkH*(n?1bw&9P;;BgAz`6bF30
!vAhLB2<pn*ofc*pQ40Yd`9Hqq7~iw_+Bu?{a)`#Yg(hfmbfoo$Qw@^9_NRYSVCvtnFn^ZnIMgJED0@5!XAcAPo9TS_?=DA>JJyh2ZzOensvg%Dg^{F=|MBc
K>LfM20Z&(-zV_f!<ug~g}NXS_mdLG+qeR-So3VdXT%6jtT$yYc{{_;Qg-Mgi7jSo#{<(NLtsYXa@@}%6kUOA1rA9`2tT!-XycqS-Y-}C3<;$no6b;hpMV)+
`&k$#@@zpFwv)b}(+0oaw9tM|bw_V>{s?K5CH(CFo~e)sK=r>#OD;EGD84MIF?i<!zoMIoP1o&EGge#o^+q_(i>^C__@rQm&Cg=$y9;vf`k5MS=_9D&#g!lT
bkRJ}Wzsn70;QI)IKM$>C~Q^<M1Qfz`tMYBwLP(ztf}ody<!MsQa8S=2_qb&J;*q$qK?{jMTcTRPn;m*X7g~<K@3gsPg53Ybhn*)7q{$)KHWLvZAmYX%jZnI
rU`%@HxIEIMIxN;OPlR9rox_QIdJ1mIz}tQc&T84vCD%m8eQBlCDYb4TmHXwlpbR;2bF6qar5Sv8LeY=D7DjdY*w;>644&opAVeDX<~XJJ=7RVf&08S2z|eS
$f7N|RuPs)57ynKt-u-9m(AF01ykYG``W&duqxMl9`GU=H=4~-MSav@&v2tPa9$fUA%aG;x*o`=5UjD64acK0A^y^ZR(w>G;mP53#@E^o#p}*NAnM{im^L8+
nl~<eKEH!dJ2Pxl-`J0YXB^)?&gMa#?v2Z@v#EGoZ1H|G&<*VJyZhB5JTOaBxjswCC+dxU2FLga{+mk2e?F@S?Zf_xTpEdxJzMs^<WnsQld{K_T;5_?q%!~Y
p9DO4TI)pP;fBi-KUepT4S|8xit|}%Kb{OcIDPlKJEFc+$~2n<p=0-*-i1<Y1XPo_GTYl>mb!lHqF^|ZogGNYvopbaw`G_&JPBX_O&J^2SK*f0*WLKLQ3#dX
)n_lXgzgdb=amGXiQ4XL7_BRWT8QZV^NAjKTt~Ed<5m@*=XX-0#1C6Ps{f1($Dlm_=lvPI9NafKRw_l5hnB3)VtNg4m{o-g3$esNbiX!PeOflgCZ+8|Npf+T
^V^Bf$71lE{O3C+CP4Yv@1cTy(TG`om-T3+6%R9;DwN1XP>?5*Yc`pJmHxL&*?(v7`h`O&hf*Q3-xpD36@(x_{!b|TeJ{M=N=daSHif-xt8NCZD?W&AkFniw
hMM@s_r>J^q}aZ7wGfZNHS_IDVx6fd-I7>m->ic09&VeSA}45(?T;Vm(?W;M;a0Y54w(4vOXQJM1~H!l-T9pU=u?;B<}^=5?w#T1qsFdqm>Ipu_bmh$3*=j<
GSYCo!|wIFWOr11UElR0aNg&=iBI(gePFjzY3TgH0l85_6C-Z}p#MEvvxO!Ni~KTq;yeZ5ITViN>Sze|DRd4zD1nfZyrhDHBSfs&f}h;>LgTacwD^nXK~fUJ
>_DA~;&_iEw9QU<*Amck=S&%vr1||==(E7;t7yOXod>>r76^&*jmM?^eUApS+#uQA!@)}6y_7;$lZvt!w5W|(sJR3|<`a=#(RLu_&$_*wluN_kh}#y=KIY=P
_%k7cl@eGoo71&(M&T8clvba_0)F)!y6pZT0*&9-W#1J=A<9xhclV_d$OYao$fs3f+>x2CQz03Q!X(_klN|9vLEGe}Y(8F166Q?Q02Ki#Nm&V3{JZ<PwPMQy
`z<vp{yC{YsoQBP{(~O6r>X{?`I_Uhx+$^riU$n3rY6%(JRu!QL+ws(4ErDN`2;8Oagz46(e_VI3@Q6@jl`XW(^PMSp}afzZ@%7%@G!#Q2&sXpdLAsE(&Q$s
XXE4_<s`9&cqqjNEB_O8gtMH?X_Dnoyn23tuklAdgv!3`Y1cBr%L{1_UkN1RN~7;*&atz2&HI-WKNoPN_+uEeuq%itUiUxxl#AX=QHPS2jq!o3EHFXL6=n7A
<v%EFp}l5hGT)&I88ymBBQ!41PWaim^wkA*X=AemD<;^tD-=j$lY~&!n}_o(lTd#vW0W&O8%0@*w4A?kQ6CUq8NH;0dG8+m+)p+T6un^c(<~Ha3WeJ~|7gK-
BV59GArt3*kC-Q!lp#@GLGoLWIUY66{xvBrhv2eEVX=w<p5%SWPda3av9j%`&4LIRzP+Hv>E?v`Py0tcbeSOKUEa@mt5RfJu{mGqF~jwbu{F24(xJMy+JREs
6bBfq^(3fWF_cD08KR?tb1Q|-T?8(=dfCXsHar`5yU)KcSoX#JM@>hpB#mG<c%}07%`8y7{-f}SN(C2{;-uS@lMyE+ty}xj8wIS{@2__V{nx(_t*^XGzZ3;S
vahx4v;>af`&EBCAOwse8fU2VqoGzSra<${50BVN9`x0vBmPNe_0<7?*qK>(7Z=z;#!$V^=!-v&%6B@w`EEe)O>!<(b~mtaSR7#tGY9AO;pFD@bBL%8@OggC
3CGO%<a#C*kt2NY!<vL6O0>Gz*r%QGOM5(rG9($ou{QN$#swH8etD5Bz#4m7w{~TI%izl-*V>+cKA@Nz9RHjbj6T|S8+lPZBrxCkF0^5S@);UqFM=+oluG=j
3WD(5)#(6_+-b-+HxrMi#o)wb0fhn`6}s(<SmHAb@b#L=PN%FNitl=mJ^E${kptap>R%jD<4>1zo7o1$G-FvOYMsHuU{{$<uZFO`-KL&15g3pVNIV~O3Jz0F
_a!16F?2x<mfFt9-OxHDqGJNlJ^vhcn59vrzjs48To0DS?Qe(7Ea6*0+WaA12aU1sqqjdR!F;s)%x}I^_@J1w)@zo6_XQ1~PM<rC&71!&YZMarPcb3oVWd4y
YD{#Iry7I%;*5!L=rLHv8WZ`i$3v$mDMR6nAtX3kU5g1E6qmYFoY~Ke7emRqb9>CNFP3_6`;{`z{WZ@DU(&#jzS&CaC|9hC`MrAh&;*lCEbMJ`7NFa*KDqyw
77`b~?iq-;!D9g;`uavgIIdAKxHSdfhQ%?%xyVzn&vPi!)o=no*WW+tIr=yeRo#;=8ig6f+yD|013YG&EuB(QhUTuC<zSH&(rJ?V`K?vJma=;{j^Iny!CL-A
DYoEa{K4JV;{eJ7dnvi=1R%>$G5fdB0B`>AEY-OSU`|insMWz1m;T;L8+m>l^`s}WM@=*lBvtbK^n?z)Zf>ax(R+c*dU~T~Q43OLBq!d!Qv!QMel0VtF+6K5
7ik_?LG!>=vGF!H0^^fvbchM>h0B%VQ=B2XsjtSFr?^0&BIf9Vb|7}CD9(xvNy49Xz!*K5kR>L|Az^X`-FX_01OYD`7v&C$XOTqsz4xP)FSPJ5F>}m?Sqf+E
&Yw{fG6HLbm3O_P8g9`)&&{koh4qp~HC+`M{P`-c)<ekC38i<1M;cXeFZsyh9%UEIll8M7wz5D%`J?da`@!IP^4ZFbMjsbQ&L5|_<pnCHD(AN|t`KOfW4t7$
_uo489DDj>z$5_v19X0M=bUir^Fz0($GY&bi_Nu6*2a0^0glEkL+G9T{YZ$AFKDXbUSt?JV4)<Wn5NJZlh(_Zv(m!Bb$j&ipBILpw@E%^MQ4YTLZ9YJese)&
54p&&xE(Oe5#AEvgp!-&^j^*8n9r_f64<W=h1M&9n`RMUxT-ZA@Yxem%H9)SS0o6zV|ht=@jU1yU*##0gyAy{@e?&cJFu3h8hj`C75hLa{R=lwwAXOTP1ZW0
%ZuFq)NUx~&)mts#&`~uRi^P;RNA=n?oNDio)s)qp4JXqx?zKhBXsz$9Ng8WPPhJcg^M;z^s^}fcdur*kn+`H-~0aWPb`A4|8$%*v85emB3zwoKE%LQRQ*Ud
i7_q)ABq>4PJ{SF^ECM%Pf%NXn7ywFM&lZt(70D7_P!lUBC)fE{%;$}W6xDVyBy9I)1Zq@4Vk3_DppwS^0c1)8-mgE%i@zwdH5F<cY->?9Rq=H)?x|c57RQt
^$Ht;`T2KZ!+K9>SDhWxyPgX7+V4rvdFwEk^LyahI3aJ3Q4rhsTY^NteD&?B2bd(h6dnXQqMx01#;!sLy8S0#&rAdm^zZU8`)N2xzW02(<;sZ-BTF5_$8LC*
C@K<~BnADdsYhqs9MD4jA^G?BI^@|!Uin~Ii}^}=`tAJ*7|JqIDYs8UwA)Lws7FcoZC@s~zo!5N(W_J^`~1LYPO~h+uS4K@=K&(00BpxGdTg_pV!V2E^uixW
+!+%6{=U@;X%5dsrre^iCt<?5>97-u_H-=E9f?NW-20roVJifSzh5geS3`mN{A=}pK{&gx&wOMn1bbo=e?*pMqOk4o`-i;cxNb||JlyGopplJ;=pJXR+z?Pv
Q`5!cGMVpJmYguIzf|(3D*#H1;%)~{*`j;Ezn5Vk7{Y=pFS>4oVK({|1NtrSV(AzS(}pDqwCa;;%2P0@Dbs3m#}OaB>4^KJ!Z1|J*g%Rj$SS<-+6eW<qkuuG
vbG#lJs4IA<4%U^nRhl`Gl9sVDZ8V)qY9QL4teT}a(KDsekQoj5j>GpLd5ZDfYsTGbB751@6UDH!lEF=O5JO?PNW8(Qft3M!{%uClA)0FDgcSXOMTB31h6(_
<JP2T0hL>KeyS1tX2PGNJJBo`F3+<693OE+;X$2H!-`y>;nFvmZ6A0%5PhVx=!x8!`5e+$t{^cjpc3dd!lbcrtMnBY6dn-D_*(A_71gPtg*&t;zq8{_w<r!?
&HV5MA49xgH=+Dop9jtI_BrQDee55lZRt8~hnpJ@GWKgpKqgX`7d2L>72u32anMG1pw|(186z}0&*{l|C!$)f?pQ;M8RV0AzB!OPBenRXD4&J{-al?HV0N~F
f8$gj&q_V0S<6<ND^I|_RAHu`oG`viPv|Il5N_*B>_{tfBOxp7^n(TmeC{)7UG5dYzVY{eJw27MyeE3h=|Ah7IcIYJ`A8snCprCuwFf9^7K}Q|oDjA5AZJK^
Fw*Tep4}D@#)HY?i_H9nP~%q=*!S896E)R`><&5L(*R>GgSa|aqw^lWDHjHn4bgF4rD)WIG|x6n`opbV+&0iz1*&q&Og|n3!SJc=XK(K_7|y%(e0$OmQ~S=&
yk_>p@$trzllMe%Tb{FTK|KY>qyrou^aZ12OlZ4EUjgSF*akgFtr3@fh-5m)8``U~aURt=IOtauKf*2r?<)N|I!OZrACYcrAn<*9`xJ+Lg&7XJjb3iF5CbQb
=)SBYt_Tf0dPBTJ7NW7zxj)Cu@k27_Z_8h0d}geT`MFE=U;mCZ^-;~EallULL`N0~`QF;c=Z=dEK{q8K+b#9DN96N-x6cP&=B10Y%;u;-WoW#f3qrp9N=hLy
faa;r(6-R?__|1B*>ivqNs>!iY{_9Tl@amnl!|~8vywF@X)G@08$XKO<B46iYpD;Wd=dV;)#KJm1pNPnUQnQL#TEKzs+(1zun7MC+O#|x3N1kfy^>Y<Lx1O6
%C;9ewrO_wuSa6_w!aCtb1qzOKEC*tz}0?XFAuco`+$}_nAO-k7$<HS8vVN5iNEF=qaDE}5IbJqVBzEg{zcUb3aJ(lI1w1!EltQ<+^Z)p4aMPO((;kl#bKE4
wNW?jG(=YHh42l!X#8y=^^bi+;Mt!PY5Dw~V7=u*rQ+>?;4gDkqh8{mDeNTX&9uhJsZORJb2acdc2q#&LMpCbdVeK#EFR?^MdU_uCb+CJ&vPNU71e7sl$?E@
*m)Cp^X*CsT!+f!**=}Y{czexzj!^-*A;KT^ji&zTzkj9Df(eR)}JNNI~?(&S8mZ35bAefu~v6T6t~!oTysei!Fseh|AVVL=qj3LTFgCR6(Txqc~B1aBgq4H
6J7`s*)D9Qa)$HxrGqVghT#4^zLb6_42~M)L=G`A5Tw}CMgBe&6>WlYVSygF)cusVg+3ndDGu>yJc_`L`+NPwS}L&7I>4PY5{PGeDJ2BdBEYdfG|wc@034mY
CTUhmNWT#0Y!VR&&el@JYo%qN7&|_})#QVR*EF)<jX7Y|&ckI|Gz1q)mp_h0_~C$7sYB?&3{d~h>Xo@}j@<RfoIDp3pd!lqDZ(R+(1$y3^i2}-Rf)%inWKg{
{HHvzFdzV%!r!J>kGetsj#P|n*jaFdH-`p%2?T$+!c$8UPgDzQz86SI!IXZP_ld^==vjEPtEu3DJXJ$3XyiknAoF}rpbMl&U*D|bbHhJ0d>l_Vf$WEKDz|K1
&`fuxjY2jM`Y!2y;c@;LD6(~aIO2v&_r5EPX6AzY-W3^@ZBs<>Gc3Omat4!{1JU>o3)tQ%@us~V1KnDo-xs|c(f8@YKaw&TXe*gB)RsEqAy=RN96@KQIX`3K
`D0Q1<mBNG5!txpAa`ltKi{Rwe^F(2RYPFvs_~V2Cw#3a7S(NV0-GF}IHU9b_H}S!$dP_E;e^K_ir>r4d=Q%?hF1>mcyM^QC2}?bg&F4!nEf4aPHeUHsdpHZ
dXhg&tZ726kza`=!XEN_Svg*G+aNO1Mj({C2riS{7e0Ko#>)AR+u8)KWBa@3(5A5=6a(AU!;1rvRMKq1#Fv0Rr&M}hbxV8`aHifj83;*}+mRB=p13zY)OCYX
1=Y{om|5h)V0FejHKD8m<{u*8T;;CB{W-<_Ww|6gIBGvOq8SMGUNYCMj##)PiMQ*N*rTPW;kaF0B0hQC%+YqqK}x_Ht7TRxj{S;PGp<jC>!ow=A8b`%=yb8r
3l!py?eGuI?m%q!?=}1-oPdWqId`0$&OuOS@+5s&5S9cy64UpW;K!1_u18cD4&S}dF=*`wI@-GPm#zlFsco(BMteAZ&L7m*A<4p0WS<mg!#RwK_s2(SoyUme
#p}M*VL0x{#u-->h8jPLv0{%)$oqAYBfB>d(Fe$R*l3$zLd4;6c`yXLfeK_dRy(nyxa;V?n1)~XI_w|!mEdKU#!J`NDLB*r@(<@oF-)@;3h${^!7$Feu=+tg
PSB1Pd?I}V5@Y(~+xa?pl|yWGODY+g?l;*uE3=W_(@n&GTMjF9JQhE8BJoo)sajOF8@b<KxSF48goEnd?@wMw5{|JkLN6k*+%Mz$sKyB#MB>iTfmyIy$eZAK
<B0a!xPUd|ROA}*J#831i>B`?A;N_*uutskyTDbAlwg{#*PD}Y!?%`bn86fD8cjlro0CxOPgfIhZNr7~Yrhz$a&Sn*Fg(L29pSOJX`Iz62{~<7NcKh>R;wQ=
P&GxO&+V!4C*n|$XI<Q$UoXbvv*GLEe1SMc-20C_F&sWIiU0f*voIJoeW2=59o7yRa|{%oh18&r!kA_%UI)2f{d}$tueML|+{)JjLwERwdRIC=w0?X~H&6xx
%e6cn;S>xy8*prTm_l%oVVqAj7oDYsm#MbS;giIQnbO_>Ok5%MSeUB-pYrg|Ej|wfvucl}xdmXJOZXg(Ob%u)^`1HQv<O*&%n#o(_`znW^Kx5wB+km(+xk|!
;mRB@i!XH&E*1^?Xxd)J{KOlb7qK~rTsVE|Zs8?lgnqj_N_g)1m;BO|GXtSLSKlL|7Xt&8TrXChLMR#8*=z_{Lh{5-uezqIIMsTnvh!sH#@QZZbOpLX#FLJS
i_H^MWD{={T#S(2cYL8T(glaFXwW*j*&@a8VB)n2!u7as78Z0S;u~FN39FMXluAw2L(Z3>dDrFGuvQ?(sf^c~f7W9A>)@G!AHL9Pn9~X*)F;m&hsN#mMVR!r
AsXijLbyQ9O})Ln*w8<3FOz5n$NcIi6$D;bw-N5ueUOM#mz_sm>06>wK#<qdFar7|z8v>$qF|?NR3xaPg<%%19_`pyctQK-rTFJ$7#m$U@{Cp)mjwE(DK`V5
WTBN3YiWVy*yL>1d1s7e4^KNJM?vOf`Fj5!bDZ=XbIaz@#!Q#$c?q2gMAaV&J3f8^LHd2kV?)lUFO`^$&<KT{*XZ-Kd1cHS1vT$CDMRZ|vxM%Z0(?D5pHf)O
3oa_1AG#+?!JT$rE35Pxrbfod8)My|>bh%nK<_-3zch~en!2DM<Lxzd2@lK|Q3cHGiNJ#4e&cP;B<wzWOe+v&4*P#~tB1aNqO+p>p(~dgI`*d~{z*?j*WvOj
?VE{^F{|X&APL13&CM%^Z|NeA;{96V!3Z#ot-99NX5hJ48Sz8UBveG2X!VwPgK6Uq_oP4wJl^(Ds4W>ouvR`+g5))(9VD9GPsZZILSo0iv;X_Qq0a=p*lKbl
^3L=3gjOXZyZo8YrG4`VcM)sY>@C2<jDMYFT+y&Tn`ArxsQ@Q>9Cug(ZX$8-&NNf9JMQV|bq{z1q0n!U(xQtEB?cY3EZQ*$mfpl&&VI~Q)R{>C_QLi-)>6tR
_Bcovp3HqD4(z()ofKLHDE+>Vu7fQd@i#axTsq_r1u~X4*70OqJRI53*I*CdY7Xb!=T=br%I9nDnT+~AiPNWa3eo@m6J5%x1E?Er8$I-QM$Nk#atrxP$msh|
CEBN8;P^&(bGj~;!`5D@S{tC6&WTMb&lr_01NuvR*4X)LXOq=K;2Uuok>2a>(4b@7CYn`7<~OGSr{yH%-D@&lA>_xNXBYlHU-p4i-?YN%PbRqKu*3MPT@{T_
Ui1s)I6^lx`cX`i3y#PbRkw&(LDqj{=*e+Ayk1M*YOpne?A?|4dF@aF5AJ^GX*Iz~rW{eTR9C3n^}RD%$A>f7#}3<>hhT(<zi_aRz%NvtZ<&={(78*(pGp~x
FJ3uJjeVK;#9~bG{<;vlhE_BhsghBwnWJ~I)(X@f+;{G4#6UW~#FD638a)_NrZ$cOFX`c}QS~7FXk0DVduoMmJ5QZ1*So@jZc3;-&IHk&+Tve=1CX%VJ1QCO
jLa=(9^s2F5L)p(b9dDa)!o(0$G-+Z(y-h+r`{hwl)_(hyEwq|C4+m}Ydw@-yEv)NFN4#jH>i6rc!Pt*GRu#az{mfRUvj1Rf=J)~+A9kk0@qOKd<<~I&b7$#
^6xI_iOJaL4e<xfLe{qbNi~EU$pjmP1!AoKGUd;@Jj@6zr5d<7f<5#v!%VgmDtFgHE~y4XNhgd><D&`AB(ZVL`<i0GzFM-<Uju&^KVSX1*BMjn>Zao~-cb76
6xHDB1?$%-@k3Ab(bsx8W!T38a>uV;H1amU0r_N;&|F*Gjw>au?l3~b?Hlu^3_-ZIZsyL{ZHJ8Bwy((wHh56=MD~Dz3)<V0pBzdzgnX?fZDo%&xQ4W}(ocn<
nExY3e1ID6bllYCeCCJ)TU=umSKJZ0)mUbhLGZQ3#W;Nf3(RTnt$6G0jf`itFAryiBjw%Q2rg*{JhjW%BSw-4k5jalBpd_q(KGj%{IVgaKF0J)#@b@ta9hrc
aQ^?-=^IK;N?KnXyj!=Y)B0hKWy9>-bq%)oXfh}}Q=bgWnDDMEqDD9tUwMtbk>Klt@0Dp@X<>|XT~+FeHI`*c7^3bOLWCnid~D7UB7b*pjW<|e%#5>pltmeg
H0w`uYa9{R<Loe!5(?p(ke7AO&2YglnjtAg3pc*hd8>;$;9%CvQ1z>N7*jh;8m?dn*2wA8ZC>6;neKJ6`#&PyGAau9YuiRaRFoDG5L7@!X(T01Lb|)Vd+6@&
MjDF_5fOo(qLLDdqJkKR*nlDkNGb2!AKw4-an_o3&CI^`KK6O6H8Z<jC<u2dvdRcYH9f;L(>Ft$)c5YWzU>3rT}~<9{tt*8mp06)ZiPX=k$+mR9nSB6`e(r8
DWZ3#0yZUy`M;99BKHLWetZXq4UhSw?~b?M3F70*)ah5>OY0zfVI<JuLORH}*o*EgCP8A@H{-A*ktYM$B>Vh{9_LcOos;K;H=n9%yNG=gX!mcOxjYD({Y)oB
D!jnZ`*NBhIRpi-gE{|ueHk+w5xD^i#Cp!}tR%!GqfGH=@5$^8s53Qb26GbM_sFkpMagh5o{B7<S+qgX?wgcOlR|Vg#(e+C;Eu_YyvDjTA)w&ixw=^zf<sgf
Clcm8@!Ex^nW{Jm-9coQ2hJs7q}%20y_`_c@x^}RQnbbGl9v~pzB+<AsYCS|eH=O-&xekx_&`wiAjR9sK$uu`DOdlOgjVb2Vh<hz6suJ^4wyM&A4m7j%ik_&
9)D`#l}GH4$s?~HeKW#?vmO4CoIzNyHSNrEG{+T%{;{aL@d%+Bi6|#T!uwd2O#5U4uDvK@NxJL@Dr+Nu{iF(FpIqnHm9#=x5o3F4xHBFT&X-?&6p9SCeGxr1
<#>~Qv+><PGX8y&w>eMhgsP`6XI)qvK*P8!em^MyC;O+lou9^`{J~Cx9eWPwOpeekw-FFMqqr<=6ahA)bao3je`IS;8#jy^qOa~h<@c&ia4(5_qfVZS=gOiV
Z9Wqa)h(jF9&ZATe<oV{6_dcw5PLQ1QWXA3xe+GSox$+>LG!Oq@(9*D!4%6K1&b7)<$&QlAnTT6g>(vTA57MHc*_UnYb&$CB?0*3EkR*@)(o|$?f-1p5KyDz
!?^t*0P-<*Ww+GTP;e)4r?e>!IoX~UUYS@y!By(=;wxVaUiVpaI23`-#Lo85{mQs~Bj=w(jvx3KZgDF7_Xw;`;WwohgWy3IYdA^dCv1C$q8FX;<bC8Z{X#GN
ym*GNt``H&vbaRURafk(zOU5w!VHsPy=-yG@t`|Nb^PmX0~|A}-&|g^#o*JtxUIu>U=4~Ae7~!QX9wHG!rH`PAid)B-**eFrk;O1a)SWhxy5T8N(3yv{J<l?
JAnt>3ZZt31@Q4|-sYmu0GHm}@5zI9pjRt;I^z+80&-h3hGQY19cEWHrt`#p8v4iN%z9v$|Lf3wh<J~5{oYr8+F^tCs)OxN2yCBy$e`#6hWH+mx!PmG$fOca
4$Lz`aPRx8uFGNYN--y4m^y>gocAuBwe&>cGIvSdSz%PIxdhER+hBuLC|ZIv7$M}w>@KWgI3WFd_Q|+1c7wIQMSilwXy?6ltu+rgxesMIef)s`ggi#=R2_i!
^bwY91$@apxpw+x1gKjM+6iuY;8Xg`j{*a<*f<b-$a$F9U;9hNOxNr&E+aDh?7bJBYu)wb2o8Y2k6j%cki(LH$l;&A9N=f-QhZV>5_O^!DnjRPL;8Yn<V%-G
R6ep{-I}z-sG^ee*8TwG_$oyuhT5X^=x(0oF9Xo4a#l&`8$*!CEL)dP1B!aBS1m3$BBhE!uVTawZh|*U#)fUc6g1#|htnJwOSApl<%2(WM+)Bds-ws87Egz~
1-icG=}Nx{Mh~-@P*z{||M>k^G?^;TF9;XojAkybIUz1xhw;xLBlO)|wsS8u21!hi@8onam@d2hM{XL3)tU%#KB{Dl5@ed#yhD&v$<QW4<%qo#H)me(5OBG^
MX#B-=gwsFNB3q$frpz*zdtMv?4w`))_)2^#&2K6Qg3JEwdPKr8}&t?(|AZxfEUVpSc>J%eDTQZm0;z&B<yn*dn9@w4s}g2N_PtV5iKqyC&l4_{}QzWC58Qw
!n??P=Zi7s9=OQ9ZwmnJq3h~5T*8reaII=TMF0$?S9sfEjPRJpWtW}J3QM^pb_?MOI5oLNuVfexnZfsa50yGWzp>^X_2ozm=4pKQVI%T9+I7#spcQx?jE3C3
m<YvxtOPR`7pSL`x@_ip!#(tXRpUu5q@I4hw5Qt*pY@IHN;5+cpYqy5Ewu?R-WhpU=l|dLddC(%=}ST=l*UMGljwaHKh)$}1flS*)Of)A9Jro!9P++t3@trf
kLL1Zs6AQyHbT5VwvbrYjr)GcrPxS|B+ljJ0kecvz8q{)GClav$b;_gkRTd!KSTyxzAFPG)MOs{G_P%lYmXfmPZLD2bj6Bbkl_SV70yQWpI*?^Y?YxO(#OxP
6$2$BKRj&Q%MjZ|?9c5-Y@8nc2!71uIPT+!hnuq-z4npV=TZH>;A=F7C7!RGznl#6GcVN#ZbsrnhK%U_?r@A8TDu)m><JNVro{m(Gq5f(iXIm9Mfrw+D-BsR
#0=kY3Vt-ia-6>B>X07}G2bOkOH0Bl%PJj0NF27l)X8%x_=2qL!K6Qx8vdvqE-4-_02@<A>Me0!_^KP72|N~vcfW32Kf)n_&PDMd!`UEo+tMW}_Nw42H2kU*
9YCG1T|05y5^{qY>JnQCDDd=~8f_uqb9AENlTIgao~y7aY>vjU^Bfks_k7Xj#E^ZE#Scc4*59u82chOY1?fX1LwJi_Zn44v@2tceH3I$MT2%k3=(-=g{uG-Y
8IHl;RC&b-CqD#A$DZ2Ycfcqq<GI_X&CyzT+1l@v7Dg8>UfEpr#nGnO@(4Bq2s<y3+z@ex=_mW4wazp+>C1I!_QZq0wLPBjF%{fv)d`b}@o4q2*SYN(fiuG-
1JB;YgJOs|%&OiMo04YIdd2}nUjBMl@yqFdbNZhunQw-Y7w*fOXLnEeV*6a%&G<upNH4g+Qstb0s<)SVmnMy%|6hvXo<^c)=@%)V4zNbjH2o*pkFf{}I>BjS
o&wfm@6X!RhQMn^I`N8(J-p40#<Z6V;idm>?`D(%+C?v<TG;xbxQz0`$zWgPan3Xb&1OORw5P*qMHkpzw>n9Y>W>QH0|Ech%HZ@VI~y5QVt%-1FMM$_4jX%V
(no(5A|`qUwnT5UQV%*w!r_YsE@`#NVQ;vEefcbt>J7o@hYV`7A<!Oky79Rt29uxG51k;^CG~rAZ;)La#`f1ydNxL(D)7POpZ}t8ZGInr`h8>Yvifa*W%feI
$xHU9&22!-#eb=+Cjd0|A8UJAVj%O*+V0aFaW9{_eoN$A432K`?{N&OLT_tyh1xAQXb-!YKP1+@%i|6ADRBb)?fzO)9Vh0Uh4kCiL?8Y=&bT$~?u^+(e`^Zf
#v%7Rn=!dDvF^d6uVT~lVf4V`^c5m6O&uDgc8Gh9->A*ujzk)?b~OX{cE(}#`Jofj7Xl!8G3H5`SPq;|{(BC$LQEc(JZ?`BjLK&+4{N`NV<Olrk}FyhQDSe^
SVR5c^x(OH^zB&eBef%yt5*iwU0?ef`@LXL=F_rGu!OgC&&aG70n~9phe_1J;k4o;ymP=Gf0XpCWo>=Y_aR|QYAzF%R9+9x6LbHskLy7xo__dwt803DHwaRy
M=JK21mp9G5*ZDqaOmxi`pLWChWf};A7-e`aAuVwK9Pff^^5;rT>Th_1S-#PE8;yLmD!SOCvwlPe!I1z*A+HnXD+2sU&3RXimv9r@sOJPLL)v`jjzIA+{(B@
VSR&wWI!MT83PBevrD_9rL!ycYG)wCiY9K|TS~>Ls|2!8xo{LNq~<&}Oo4a_6ub5$;^zjJ!qBn@C|0IZADQK1Ug)v^k|y!>6*>Mq@{5K`G^0@ynJwZa3%%aH
2>_Q08MT122a^Ai8wYTQ!FqrAeBnqYzJ8tXdHvfDy6WPDtIWjD-EsJI+foj+b|l$zR}-Nf(wQ?aVuEjU+V?J=j>QVKP|}%!5VUnLy<V{O!04q2Ax{M2hUe#u
*%?Eqn5t%M5zn!Xrrp@#vx0>@H3R)qIq<gLE}Gcmi|lhe{-kwD*hqQzM9<$F{1bHdD`XP!lv`!)tZ5W#Udd-j5a-X>^37$9XkspQZc_NTZHITK8b=B&tw5ed
TiDR7iLK?Lb37+JKt29?cZWm)qvg7GX`kgF`)z57&N2a;^`nt7is{g6d|5@H%Yri9u>eLYd+=?Yn=#;5hqcI2+3?jkXu2Gzo3L|69~iRqHj+S7^r73p(gkA%
)msnmy26AxJ2{2Y7hmj-N*fy|VqI<TMbR<`xXq7B(3VDn-9fE%EMEbxK25yrXJc_CN<pR3!xy<}&pa?F4z9dlGl^Vh@TK&>j2$%v?<C(9wP7{bi(agWVIUx$
^<vg~fhoFqU1lBF{jjY+dZAZ60dy;`PwjhL3~^)e9qABzylwNgbmg`H{oJql)lM(CBJaOiHan;YxC!K&`XcFvhM1@wFUFG+Uxp_oAYS&n(a8^SFzGZ+%O&oi
YcqSuMjMG9(Kehy)}w)%)psJLEB2@?GE3yC3PaQLH$}7iiN2#$_h3(4K92mGxmwv{i$ufeN7rNY@YC+%@bT#=P|ld=n@Gfh5W7CNeIgo^-5GuVEOp@imP_^c
_W&@T`c$x}5Q+oF9Y6C=hvPJL?r7CfPw0-^8amwOhOG48|7xbK5PyaxgKJ+5?kNtfofavEV5g2;Mv}z;`28;`qrD~~60eiVMnqmZqnXtBUGKeY^hx(N^(MsO
DU*gpvqvD7pWd%_`H};KUeoh!a|NR_?*)4ceJE`(*0sM$h3cH^o54YIy!>869!%i{Xa2}+)`2L<Z`CmGYq|o$9-{#9zI;S=3_c)JC(cV}f*IEfN2K-WUe%W&
dRDj5kIND6*h@iWscW2!ADZ^+;u1EH`;)N3Z0U^BJtUf6Z|mZ)Eo&ohlocX3_iP4x$lz!j<2cEt545%Rw4Un^z|)|}j@b+XNU!mlCH|1Z*!$^BV+sO{ZAte=
$n%5gh0qH%Vx77#HCbFT_klpJ`O9Mwp?LeASBdmU7X;giS=KYHka}gQy<$2D1;<J{q&~SL>6*MgjZi4m#u;8SNE@RzkZXZQEdb9SliX)JXAG0in-K>D;_&Rm
-e-mn_0gZ~e~LCp0Ii!83el302s)Zd;o6go36<d*)innQtI``revC(G1MS`Dj(9B4U7XpP2mqzTn|66gV<-enzI;0pgWuUN8f*R|Ae`m+PZlmFn1mmZ&hZSx
qR4Z>Ll5K;Bl`GlmY6y+>WW@`8Dz&Bx~uHZ^3BoC=>F%zw{Wl@%$64-=J6UcHt7*Q8~hsQV@Wr6fWMB;M_r=lTz}OaaF)aXt<)quCMCX@m3GizwebZ--lw=!
7hfD^zDy}{#0XSeT<(f{ol$gia8pq)6S_++FL%ZwAocs$i^FPqcyUFi(qkV1tq0rM^d~g2u)pW8pg;`ndiEWft0U&5-2J2Po`!(xpnxQ&r#-gscz^0<U`N!S
p-c&`0L<3WKQkz_#TzCrA)^z1_!FPHu(r#CErF*a21K5EE(x2wr@MfrC@JTPqZSysze)J4;DyMBh8h(IE9CVGShznhLzizHYjLL!E=ZTYDb+N>(SF14G#+l?
4pBcbC*qHH>Q+8aS*sCt<m|&K%1bC&pBY?nGewzKPFzS&Ay}NeE-i>85Pd<h=E70|JXE?UXJ#$2*Cqen`{%?QqR)P7-!o;D9KP~oW?2<%Lc?T-C0rn4`(Kzx
l{^+1vhMwD$%Rzy_!ggfIPTlmix#$cLZiX>cAT~o_{!Ub6MGyGDP3qF-QY*$>I#GFKYx(r*BVVJCnJ0?(StiL5y$1$+UH;Vzuz0)ypS(fM9hUbB2u)c2}#&h
$<hAoZ4c_HGR;V;0DKc=7_(C}L1vdG{kA=E->H8+E*_K&x6S85A5$+w_`@7|$XYJyX6fDqjd`F+l0IE~BOT4=Dq=BIrnq}Qq|%&tp7uWTNO1~VBK6XfO}~1e
E%3g_BY8KdudtI&5Od35(s#$vTSR^zF<cPR6ol(nfwz%<wutzoAUHeZPwdB}O9wy2fKBa$z;#<=tlxdPVRBv<(wV)AuG!WYp*_E#;6wDF{<FvayvqgY9ib(b
RAS%XmAZc6Ya}*9r~i&}M4_aA{?;osGql+L$e=S1MBXj0XR<995fE+hl4{HsH~j?9d}Xo0`qF_RlWYT6a{YM28c*b8m*yi$-eAxTK6!HU-!<Z#d461BB<9Lm
(U1rKyucb}?q~498%J7RRe9+eL8RHYq199wlz;aPc3#tjs_Dq>GcrctzQ^$6P+2r2ZLI2>Kc}GRxKYVFXJ3dnHvU=4H3ciP!pjjN=lrKGhtY+FfH!kEKrPi5
b-jA5sT75nE?`eL_jbpgtug7sk^nHy$R3RK3&n)-#iT0zU|d^{6?%}#1*Rv@j7pCOLv_3Cg-^97cAcbR_7IMvB5l-YO4STfb<|H>dyFy8=@nXk%NIvDsIy8O
%ds8uiHvMM5ov^h^99U9&_DL5RZ*dWxJNTv|1TfQ>DN2;XZoX6_OIc+Qz0Os>Ac;LXoVTJVud4i9&qU8ITvkC?3dRx;?6e0ptHIorP5}O`i`_i$(8<iQy=Fh
-*1iI7XyXfc81}ZWEN@etPSS(Gd>xAZI8JxeT4qj3?yAQ+rO)whqfI3@Aq4L;D3PgP{&+3y7-5lUJ5eBd~Imt;Z;+-ZBk#Yw<OLzb#M$VnLB#g2e4xthv~w*
XDl-!A;USuoe<lNfZBU}Z&=(Ad{%hhXLCo84q3ko4vhr`*?fr&DFKsZJ54La_Go!<YK)YHK;*1}dB@p8e1CTLY&V%duC20C{Ta=`PpL54f2OGzxTfy!{U#FC
Q>UqWNIdaucO$Xohatp^r%!#ZARv?KabE7xD{vQ$X<9XO#ZGc5*>y5kn5$o1C@@!pOp8fCeVH8s^F(dK-m0T`zhk0p-f389-kNy($PvNkWh^ZVBEdTG;p;uG
La-N0aFR<!WAE$-<<?qnG??-}9U>`#oorMHpG^|(Y3L`!_};*1B76AHR687Hes+3;M+V3`rt3%Mg-#h0Wv)d%l;&K&S)rv2rO5jo9OYj4b7PO_E}<HospSi-
t3J4G&iDMBuL`zzFWl-rB8AXnDdFvveo(5NJI4CW9(k+}ZOaGzP}5xegMqvPX+{FacI4vl=Az@T|1z{8{zAQ<P*8?A?(^yc?N&H`HOTe$Xg*fYK3?k`JC88a
hgy-8p>UehY@^X}#<ud|11eqe(BtddKK8~CvK+E&vVV+m_MXF*);?Vv&1iieb)pOkXFm0`Zs>upIoxnyiGUQF6Kt{ZK6tLUKYuRV8wdXiZSJHJy`|#d)OUAf
koy+Z#OfuXktX)l^l@U|6*W4}^~eR5{N?2>e<bl+nsqNNi92@ck{EOQoX{n0O5>9s2q}X-dPgb)G3!4d;-M#puu}u3-0Q)><ZJ0zGbb1pzM8yYYl8h^{3fqz
-s9JoV(FM4CRnKAA>W?S#Xu$D69v(qVs!)Kw@Llcp|&zad(Z&%7gug=68A8*nDZVdZ51$e2<~eP4@GenhkO524B{>nt`B}Tf>YtySI?q7QG2t<(nLKT=4mg<
X^*?$n+~OX8jUlSPW*~m{(R(r{EiAb#Bcu65hcH>e{gR)z<_PedXQZo1HS@qyU6FGJ>BgQ!POO6`d%?X+8+3BPM2&rW`GOjHh*lB)bPWXcj-caEt(Acd5_d4
BeDMp?>kv%6ts-4yeaX7ga7+;Hovc<>2&ag`;DFmcuB8CKB9y7B61fOO`Ji`W%cG|m_Gtd=#ER5rsC%xrH&U4IoKtUe>bh*04dK?vClj7@aawS3+Z5M(CTX!
wY%Hn#l9$q=wC9JyKM1<Qr-x>?`$beOI#s0@0*@e9*p+|0n*lr=U`s9>+(b^5p@QP_ciA^FybsZc!Sy&_dIFdHOdfB$gO>fKhOn_dSdDbFYWQOu}<w@ni9@D
-(exQHUhz~o1d#tA8hiNx<TRv4*MVu=GW)pp7-;xB#i}XNWK64snUnY<eC5*jX4A>Zde5{5dGz7Ysl&WIe7Uom^J(~hP?2-oQp(m4TO=1#?KodeM_AyazPB$
U2C<^d&6;-cEI+}t`u}m<*f1c>!SWM)#vRT7yNorTTHDS2;2ET&T9onurKY<j~ezw6wZfNIMJ)%i)q?oLBAOkk2ien<rf9}&nhjU?<%N{;uK8E)yB~W8TpdE
TKKX)R#5lQ3EgEYZQ-TH*kt9t-A>FUk-l=z{OGNbSX4aRb;bm7b;<&_mmP7OMI)#2vp-C|-4%)21LI}g_Bwwrl-sNcwI6VRPeIRl!zl;s?uLq&F&p7h=Y0V+
6-{_+sW_b8vOvd0eYcm@`Z&N}KcTr-7czS}pQ;+$L9;jdoWGze)VX}0+;Zf?*PAJ7T8p-*yke0v>MRGf%r9*NQJy%Jd)<*EkQ>?m^dCiB(8sefa|MSK1@s>m
7w;$Lf;Zxqy5^EKaDwN(SDKS9{<8OJbctG`nl8XTwZss&lt#X1>`FuQ$X<_9BMVI5`JF6v!3Y%>vM;GVi--C@NrMB+vFP+p@-F_Tj53y-Tz2(fbaYf-iywCc
=Pz{^`r%|)*CmK#Ueto-&)F*F>WhHvXwUC|A`tRA@J_x@8ow@o?im|V#oO$Y2MRs8FdD{rQ;Aq#p&JuH`Xo;9{+7Tk`^x~HilyuMvJS}p@06$Wv>j5nP##Y7
M|GRptkP~fP}j&m$Vjljpii`pwUQUEMdd3<aq0eVPGhBX%r+`Kp-0dgTxNELj0NF}b*C=8Pwn!}J+r`RerGLNQ$-{k%_H%x)y3^8O1Y*HbBG!E={4xaV{`g(
sXlEmMh0(2c;0ZrwEzBPu^ZM9%d6F?i<CfR7vG7MBs)-e*sxabHA6G^!oK@I%<;{uUhK+Y8@RM8_y_1DLPeo8ZTXKay7=1sX7fei%f$aFR@V`XU%u#g*#<-E
+g*W`uR$pKo4Gq#=!Kex-c!p5%^>%aB;nw9M`FK5u4<1*LI1LiEuVH7I6k-=uJ>5sY(OWgRE;0vWZ6=u`228Z%rjxb#Ri%$*sAZg`QZxfbjR6wee|+2x^}-!
!_k&wKdBQ9@a?yNzAUK$3a?MK-j@%A*@UfBf~X6Im$@_|p54GT6QSh(gTC1Ff{=Xso<Bt9657HQT;MySL9<a}gy8t#Gd*S6IFv}m|DHh~x%}Tto0%;kJh&VZ
=N}LLu-g>56Zt6by6}e~G642*-c66PjIk24#C7NZ0k;#MY)4%PMy>hQNt>Klyq^rM3(RVT662qdgU5p)nCE;lp+5@TjV5JdD}D%ci+My*=LF5ERUW4)DR5}s
{whfo553(jBc7kpSl#<4D(&WJ1Tt~2yA(KKG`(_U#Z&{Wb1v~62W>$u>Y8v~>l#E<T61-+V(~%x`Th4iMIhsQHgwFj8rrXzz5X7IgH5-x;a>G@#Q8T%FOVie
`Rr%Pvg5@3Y$~lhqvVank2z=fIIPe}`JR)?OB(C!Mo)gm*+M4Bt(BwH7u>qTmnEre@%{Y0?}-om@b`_UE@_J+YKjIl{mM;I`_hJD^Jg5C&%L2gm@dU;@Eth`
V*StkcYn8_sTnRG+V;3UbwGz|CTUceH6*6IOZRkI<JPC<gnQ23Ab+WN*GM7?Z%s73G$j0SS9-$#h0-<L%XWFLc(5FqjAgb-CZ3QjchJ{LcfotEGa?<)k$9Ur
UajWs3d<mS)3Yz*Aiqgf;l@`BuaKM@Wb~2vQ%1%i+WZ{Nrw?718mfgRiQT~KOGLk=?I@;RFor~uNi1!-4oXtoPWR^$uo*FOJNjQCLM$uwrZ)+oUYZ(XY$ky7
mL7NJVRPu!w(vK$*x>7eS&4O%Ha<;|*>M$#;qxI``{ac<{7KH{uoQN}lgZ-4Hz><NGH=D%ZtRP(sf5B8A<<wnS?Z`$4#tD)yzJKY;Ygt$@H>(~oUcjh!VV`h
#0D3=J9okmhH1x6SZ)yOLwM?{HOB>&+oLb#LWJ@4tk{o7f@aWK+PJ_<*$mQ$$|n=<SmW!(2}9PareNy3TfjOj54D9(W^+>linda-D!0wBP?)KyY~Tc)XLB!_
B5tCLCywgVvMnr(KA!dGu|&{6^`F670f^1{Fv9yX7{>;EK5{;(#FGlP>T)G<EX9!2JBixhQkjZedVoI^eFXwzthj*eKP}pUh9E!Gsk5dl2KM$mXUa)2_~m|U
N!&GnGi`EQIGZT$5{kJLY%FnT_Cn>LpEsu2Xy~Hd!x0-IH8}lP6zziEGg>aX2+ojO3?=$h<E`YAv;pprXU-C(5HW#}V5Eu+anB0v(7UnF_+yvDgZDVMA7<{4
n4g^0!Mz>ccbclvIJO`~nItL+i>qH%FS}Y`StW1Y(JBr(W8rU^Lp5QoV)4UIq6%eo6nXg(!I+cyu(#-r67Ja0e>szFf*<K)94%>=aPSfR5Gko1&UMvwJ$<7B
y$5Qyk4<a9Sb^_KQiumsr{r&vOgLk_pswn5zdYn;`kpr!$74j>Pt}c08xLp7gc#_E`RcV`hhv8lGGcW6_3HlL|G~?gk%9xU9{Ms=$7ur;&`m`XxO317r{C89
Wo|4&o%Eq2jm$pq*5rt36?TMsY>oDpSqr4?mwstotOhNg=qmDe4#<zp>Gcz12e)+YQ}eT-pbv@cJ1Uz4d%N{pRdxQL{VO-{Fu)tH*jV=d%=E;1c{In&9Y37;
eR;1cs{>dIe-8+kdBEjmKZ1u6p=qF&_w8yKT$(wi*#21~dn<`r$0-s=ZYYziTuz2zS%lXGK3B9z^d`;UPsRt%welMT0`^8Y7VCVdz!G)!qaIFks2A7kOr-cC
?i^igw5kEN1WbOa-17uS`(cM4p&_6WIw!+@o`9w+>j89iMyPsIW|P(&39^{3QbXcgM13jEzNuvgtC`rAxne&IFcjVR+A0Yj)5TWhaC>lR@Gn{ihlA9~h0J5T
4yuzS?flDOuo00t@`FMdj?sn*6_YhMee>L=z_K02!Y-0atVV))-rL<TNepGmMfb+8n?r(JsE&nE8%pLg-@Ia-K-M)-6eOO6tKDyGvi`*&LV4Y_PM05a51t!m
irV4(?z7B)F%CGkEp$Qdts4aEm&5q?E1}1qJGxfX13&77QifP85UJ!;AXuY_#_I4Xwy!Zr`71jrcF`Z7*!OwxEGNR^=1MF*cL=t2T+HkH2{^Pa5X-2O0=k`Y
S;^)SP(9-M`<&DhVFFJT_MQla$~Q-+OLx_ANq0v7X^k4LJC1534!UFMLt#wnVi@8HCa+$cbi(Ft%J^!qBh0Lu8&apTkdpa8Gxnkz!kYCOo#s>FAZ7lEX`czM
tv_vPc##W>c;SvY%0R3<b5*2s3&+9qwjX>67r`FjC$z)l0G8d9k1sjx5psTo!717S;Ub;2-}ZSRNK4>8cc}|nE=g-QJx+k(z$^c#A6`h=OHXAMZ;v(^T@zB4
EZnjDgQ<6p_`JZAE=}c(+%%07|MxwnD;FbAWS(`v!;+<Qt*kyEvHz+>YZ3wVxyn=}N<GZ2QwK7KyW+f1Kx|ZtF~nMx+dn&4pf%t@McTAGXqDLx-7)k9DeV<f
Ize+Z#<KLukSKyhJ5W|Y$QR!vD4Yas<KUAJHNzVd0PC&s_v`#gKp=%?W<v?c!<k(s?}XrHw||t>en*UtK77{aVF&N{dz}Wqy#F_+)yj*c+NrkCe4V;5eZU|7
zN>5&L@r}%Z02iCAg+4|1T~}?!DeAJt<uCDd$KgfGv3RAka;ovRkI(KiZ<pFczp3#>4Ve$x)Rg}ed*aGa@qO><=3ymW(X^Y&G?<Ij=Pm9Qr}u*!E@<_O55#7
)O|RR=k>-Edfj2gS26;iH&K18_In7TGMU<FiG5|c^+4Z*nBQ*2v;T~9EJB;xj&uxDAr6{cC3Mr~ps&NsJtZ&z9D_?#?Ya)QQ9PH+cT5+&8%n$LbXCxu_;@Sr
T|C%VDALl0OL5|Wu8vVe8j2lDotsXk<A-ed)3EeR==tf~s=OM4j`p~qY~@IJKK(J+Ce#3t+jUiBX?}R};(+smq#`ixObot?a)LS4B_~4@3#`8QtLfO_i)Drf
iqA-ckg%TIqj@h6jq$_ln`6YBBGWm0yQ%_pE{C4+YsTW|sfWdX^!)IjnteU_yH>p4<h-{moQPIEzYj}ww_r#Uva6wy2f`DY0G;t$Q0F_0b1Ro{<@t>-`!8RG
lfy~V!(;@o3YcGbq*IEcoRd=_drC1-W#*i0pMvD5BAxiGR6HV|zgAwYj^KUAKfSt~h$a^qCKvw{2x-<H<YNSW>4{aAe2Ryaob)Q?#~_TkeTaG79*OU)9OLf8
Noac2lPwXR2<vgubcYn;d}OqSbT2r<gVdv^lav6blT8f5dqXjJWT)BrRS|T0EW5M02=JeyijPvNLa<eaw;ro6$h6wsMz5P-o>HPKmiGnv&S`Jj+Lq(9<UQTP
K81*kw$WB-kAj{_-&i$O7>-hTvWOJcV>Bhx_VMvryvn(7H`Ct{1h(kBPK7d*>GWEkJK>Kqy_o%33qdfNp|I$iFG3vOX_A=3vv{Dhv$>bM2Q#TjcW=~X;M20-
qW*_Ugfc%WoXS+gyaQifw?ZHyo*(Xd=~#f`fMYkB*fR02yn?%dN*4hGZa%5Gmk`&!P@O>;i~2*OZloTOFfuY$;SRWl;@abUp7R8pylG`~{+JUg$;DT~Bge4I
NO6+(O%Xnd7BO^%jKW<gM)*v21ccw!KGUK!N1w{M7A@^`Tvv8``SwaVZbkE7VkC>j{=b1T;%;TQCpHk3-4}%MTT}nN-RF-|>%z8xmIy>t`Hyfv&BoAWgPVpQ
&GEPS&2PF7HE3oIUsOI<fWuOey#8<F(X-dSPOgrCXCIYVHU7GTKifS-?wJu*pLfaW8CoI5XkdfE$QK2{S!cx$TY-(OfM)-iB6zpB@|zdJQP8_@Y^c@;BYWC5
ysqXUll1(v8@*u&Y+?`yRVU`N*_tO)(pNB~ow_(67zF!kT~Au3>p_<%JLcwQ1rPF2&e3%<cqiNFye^GF_x!JYzs^{oY2&`-=Ji+L3U=&EIUayqzTbJW!z!q(
E)UpqJ`5B}BSpIHx!BjDW}G(e0MQF&xxG@pU}}7QMYY@jwn5xyrARF?7AR?Z;<W{)?t)4ChBBVT@f)`mso>Qm^0C2$LVQ@^@|xDngP8gB%(dSxVC>Um=z2rk
%l&qvGRNd$Fhg+v?ct8c=g-*a>}KF=x%P2iMlR4NK6?IBF&|zL1DR$Qn^C9thl%l=HCSW)BUGN`Ao(9XKXs7>SWY^~x%4EUb%sV^yfqvR{yyW8S-lt<>&Q($
rURFvGWVjJe&C(v+W#=Z5|`^aOs|NAgTV6X-JgCB2vjRmT()pTLgnrkO?Ej{^{+0gl!n4yih@yLo|w}&?(Lv77K1mkJwN(}AWb&P^2rf@bdyDKABG*;{yXaP
FFg<?PA0ZT9fGl-=wScIRq=oPc9?NkB4v&M39kmd3I8R~1dH6|ShxbR_n%foOqx)_8YC0w5Q_TsV<Z`6Ihd5{^Z1n(4*fkPdzjmzAjsv{K$s80(+x4h3hFb^
@4RVjHd2C=)x-aq<6IH3-(Z;ilnb8w9Sz9|h{sR*^p@NQG5FIVSr|bcN6Z1dMuRz}pnNTQsV=_)Yf{CgwNXLH)IVFCt8^7HN5hT>PMBjUG?aa_@G5@U?a8&V
jz(_VR`XrcN^r2>?l!v9hNv@@y-&tcG5V}k_0NtUx&u!7NOQa49{);@N1#69%2(%OXjJie?E(eYcpyIfR@1X;>A>}&97+*^I83rqeW?5F3fc9#uC|-uu=n?W
Zl*y%Y1p)L)YAau8^3L%IN}1)FB8@t=j_n+P{b%qDIVRgAGsg?9f6Jm*JsS`S;D<q%RBF+Eoy99Bg%#%;6t+Yw1(6l0_A7_exve)QSaulwABc7(~>=ZWUGP}
4&KVY>xmd?y7KzJ6(_vwVvB3ED}wJZ|C>bb1l0H#Es~t|#so=e3#FJQJg9kg=5h<czun<;_Idz(Q%r6eOUGli@5#O189P)m>%B8mC-P(=gZ7}mH|D>4`_r(-
g7^N1>NTEJxF0z$_iFe7V&>dG9;nPk?Xa@OV_E{3e0AL|w|()PnYll_$rruPU9uI<-Gppj(I2%43*2_3Sre@q0*hh_6UAdUWN%xNkt%xQk?H--OUa?wGQ30D
>EeuHGm??v*d$!rEO&S<?+CTfiQ6B}`-3Fx8yC-y8+gK}Q_v#bgjrf8FAtLhe5rId*EnyF<yR>j0!A+I&vr6!4G4k4X)O`b5j))Arck9Ajz?^kNlF=40y<x3
vuzwF=D}>0Dw4y2h(Dqg-x?bTX9F*Hp6yr&2#j5=;Yfk*Vj6|v3ro-`elFTMVvYB@3aPPARiGMv>-00eK<KOOopOkefK%+yw3VebLg`1yv+u{jFSse%;9dxR
s=YX4wVI3luPR74Mf4E(7|i^#iKth~GbtZRfxv{aWt*fuLPw~2<OPE<ax|Tlt;Pepc@GmG9VK8?qlhDrCIYW*f1ev-A;4&Tg>Yr97&{+zzLZYq;v6&8pA?<T
xZPqOrCE{#gYU2Te}4$VcJWX7e<wp>{adV&n=9sja~iMM=&ZOH1s-M=2EnUgaIiIbG+^X`6}r8H$<=PSHm<UeNbZ4DrX0mh@f1|ucw`#%!V6gzp_gx2WMVF%
Ic;@19W@$1PJPlYLI1{7PU;139MAsf+4;i&-1YabhnM>y-Q4k8_v1@YzuPcIE}Mmj$Lh?z6NWf<Y?DU(T^kIF&OP^RAVBekAJOFl@J(5>=K|3eNoH8(t?hhK
%_g0`YiWawzXQRi@9W{RSpAW|a#?swZxFR})F0a(SFRdz_`-7Pu<Kn4U%d3Nik%GdfRXT@CrqbZF-y5|po}=5d!Fpub4kw{2P<DKeT(;i_i?VPr>{qW`CRbx
X_`XhtNtE75f+PW;V**i?xCoO&o>Z?@WiWSi~6>Qx>yTk3-wd-hVwZZsx;#H?{bevp7~4ku$2ss9*JzIKlGv7GZTdSwPcH}kE{?9?v>ejHwXo;ieqN_VQ6PA
KlbWE2ENo;Jd4;30=WdTTK2nwuBG)rfk!a5Z(Q1c-p~b^9x(*PAp%5KJw+(b6rrzcsEs?vAA7sq4(@BL#NU+ZgJwF`FpAJhUYHKV=GTW#w+!sjeORGb#Ki)l
Cv`Wj2}Xi*;gwN6Z!R1}Q@8RMHF46rV@kZk7gAJ5-#6FlgK0xR+@YZnH2clebxoad!T<G<{XQ-rc|gm0JJSY<1xcG$4nd&iF`U;}a>A^}k-mkRbmY-4DUzN@
#I3e=feZ^T><GTJpZ%VIs%@U>Pf0Nt(R$CeH0J<^=*o=(^*EU8x6XtK7J}{-^4F$QaXr@TW^@~Ie=r?-!!Z?!7uWvW%`Z0w&k}*<P=hf@WI9YF!(5T2%o|YM
n1tce^<m#MiCk2a38!C9!ldydN$W$oh;h^5J9AhM3_|OY=|r!6J|CQUw80hqCKb<x`Vx_T|E*BWWo<C}?u%vImjKav;WKk0F4+HRU!R<nHI7gYj9)ERg!b{t
(NfMV^f(?9q*5M%p!l9|UrBP1sLgF_!%y_1MzVzKF2TrC8SRw5<cf9|o!?|xes~c1H9(x-3Hz(ge!bRZgj3}rgKgxY__&d}wwrAMfis>L=Jq7u-=mw><rQ`4
*L2FP|6v4a%QJNwxBa0t!O5yr<_0$w-%A1cX7DF(dhKKris-XK5}<cNh*#XSRz@_g4MuPf)+{lx@@c2_A<@&Xh4}`EB!IJtsY^FH1_${&G$ozA;PUp6Z>C28
o^LpRDPZ$Qq}a@>f7gAHFflIbASQ~#)rx60+e+xW%d}Xp-~*OcVVQ86NIc;rj3+(uMOt!&=_}(jxLY3jl1<?V@yC)ZQE}GDtyGcDvh%^(?Tnx|)rR=JwI(`s
Isg<ARD)f^rr4EMbUnM|j?w>|B+XsI;F`@q`|wu~>M5HT<&~r09li6W^N}#hHfs0pd~w9%V46Jf+!8Ds$V_g}$)Uo`iIS})4&&WjH)+DGQSUux8#(NTMJXBM
KiW<(92^|B8c@Z5^B;?Dhq_^KgJ<FMBWGMHKDBs;hRCfm1;^b!I3tTZE1Z2f1PwvASr7AAWBFek6(w^#P6d&Y+*q+i`KQBIa?hWF-z;U>s(%UO?KsaUwt1ja
Q;EBf$X9v`&I6+^DM)0|eb~=ri+3C+zumeYfz1l0GMlwv7{_M!O3)i1$6M+6!a)<1se3y<dmRieONws^m;BHxe5*VBcR4Kftjul;sH4@?r}%6Y0fvVfxhuRJ
;M0+iQT{R&3vVSHBrJ*Ni<KO>(i@2Hj`xl`eA5EYgW)e*eeMX5aXv$lX$NVZtaTAqXFR5L?U$%?fPF^{;h3D%|M<<*Mv&kMjs^DygOuZz9ik%{>K|Uvz{FI&
VxNdT6!KNsdBlA%f6+Ws*U=o2IY$(!21~#tQRQCl6a)UPvxM@kG6?kqJ~lXV8JQ-}YhMOAqj~Mr>v4{Ykh?!CXP;mL@6LW%tKdWkg!HAo*inX>@C46I9Y6T(
(Oc=`Cm?I^*_+o)HZVWP($r~eh_icCKcp2JK(B2?=hU()EaQt`<h<8{OK$w->QFs&44Kkx3aZ0x{TCU{3IXI<KC>UyY!Mfhx82KUiLy)T$puEIakJGl*KOAb
e=Lq)u9RTM?W~8qNjVPaOR8g#_K(2<E9TI)7#HX-9-+vvzX@-fK-JHVde{;ES$;3c5;RpzA@&?Zj&=ULp~UNp1|LP^-Xq5NCdSgNf6D;!;+u;f-zk8^i@RbW
(gD;p9`d~3)Zi9eRKl$wgqe7y^etXBl<&yx%lztxa;-*THZcvfTX?;0AMt?5W9_K-0wFl|OMhLX${K4DPoh6jn_!wwW_Nqh0e4TuwLIz3gYyPuu;;zw2%N~j
Vv?K=M!5}xYrg<Bd$HAnH>5DApp%evnH+KILDkM_3gG>{KmF)QeV8tY^f&ycinP*QyWM0NkT)>&A24voG$*T$`->cCg??`KDU?J>{#Y`(sv2yvLu0Ne8=$~0
N=r)151ijBB9!h6;T&zznV3Fezhz2B8{etGE>k|wN>Dt~+yiNIMb+^k*)xoy#Sr=Z7a6kTH1Lz`8+~qf0JH<{YmjXPV9@D=+4wCNj88TyvlKc(fN5Q6o{k!9
S!;^_R_#&uk#?$JNeeZ<>+M^#9Wb(yXKDKNC<ya=^)Ct8fq6JXzW?HRR2k78tPRqF(_(oHp+pnTzZt&?Wa)w+>NZF@V~kgEk2CTk0>K>}?>NCIkIc!GaC2jI
bkW}5G)Ym!ch(Hvqn8r#Lc-{j|Ic`|msoncp1X)5JDFHFp?r9jI9+-3I01s~N!RTHQxQe2$5^o{gSAZ`8Hrj`Y_jzmrah3rnGvVU%n{OHHh!*Rb-@sq*SZ%9
l+P3Q`=|l;WF_9sO{cg$&p@V5{;%pBFUXwy@vi8GJ;a61M44RE!}^uo13JM5u&R8^aZ}$Morc%0`_xzArkmdREU$RHGiVN^ueSf+oC-_4q1-Q3g5jCtdI32x
7{7OQWRqZwjhqwC_lB%7SAOi{Ej>-FEtr1}erb$TeMbwZgRg=zq3vPN!D470`u?d=un1mFKik?z%%M3^p`mK#kBC#5AGVe+V9A+EA?80LRODWJEiq&Z5mmvS
nG!Qx%2D09cP|+i^d5@^q~_v>b;!!2V-H|NVrr(IB#IT@7y1J2u5k0(<Iyi`jK0zPbw1y%Q1kZmmc2J0s3U?KhMbLYP$?xnxJ3q12RNnX+KHT`x47|_!U;$9
=yjtXC}8i^VT0qQIvCnKJ($vLi<9y9=~A5}k(O_HR*T33de={v<(;J<xi;-*)n|^<Av#F|E#M7(e>cmdFm?{;O-2POV`S*_&I$)RBoF`hiDSnGB8hMGek`jY
aGKis$0I#V+-MZLzhMo1mmT_F3(ANu=h^%B1vlK{B{+NT%7K}{Q~jn<ig*v7r31;0pv9}Vw>-)Y1Kj-_`tfp@wI&}>XF88u*>~!1noLpe)goX>>>uj$!>@L4
T7uC%z3+IV8^*5x>fL`q4}T8?9O(F?iY`L&msu|@{7`THFhhL)VNIEPFaNS2jaN9#{G1L>RH|I;v@%A6QoKe3-#$z|IzXn7D2?*!2quwH0rdTvP<zJejVdnJ
mWXSfV6hPF^~`pH-7v#PsaL8*E=SFaeDK4QlX16Zxy(_rzJ4)|(iJ^B$C{sYJKzX4VW4AG58<aRTvko>agFbFLfH{J+zH9-U2zr$3!8%4!2_~TKe1X^k?MxF
-n;Fi9E$idBDOCjLJdJ8N>zsay4ZVXl7H*H9^4mh#akcH$3WG2Uz&X?0O<(=eX<V*i?&NVGCZ(JvovD$(h_fPyRT23)<o@xKJV@lF|glRNa08_f^jfMq#=bN
yzeHTx+SO!rQvMD?Q05P%<FwVzN`x~aq@9T9$BpYH4ZjCV~JUnHI3S%Mo?If3FJGji$mWCbFW;~VK4=ona5VR<27lMKBfn@Sg)xnK3n`6d(wJh)eL9fr$_sp
F+}lNM~LWWEd-w^FVdltMX&VwIgRRz=)2i-KuMOEn*<HZ*Yga)tk0|8u<Qumcn9(DI}#wT-7}{8&KmpJT4^;H1n^&s^NwsBE!1kB+XwEmfCpD=vrDBP0zz!~
CsquhlXS{~?YtaH4Xl(S7mShc?`YFLqTe)Cr`>p(Adl9DjVJDJ?hrlh!2zdEN6_xoTB9%%L0$#V_|OSm)Z`K9t-M*l?aAz%zh4jq9(fW9uT<c(CG#fN%M~Yv
tjX{2Sz)99-9W;oK2)^7OI+`gg6J?AgEzA*lKES@X1ny@7E0L{@<R)EX2~Y6TH9k$V=%k)vLj@Zw^zE)3xoZ)itR6nOL)-IZPIX89)6W^f4kUykXESMw3b7Q
{Z7sm&o%(%XO}q-cIu!i!`!s5K?I)vOn)`8+u<0AZ=|J&0E+UjQ{DX`2D1C4r=F6!Avn_@;gB3R9NeuatXe!kH8jrMt*VFZAAB(*R4(`!KKCuwzzEs{bG%<u
g^*vul@g)ciDH522SLvzF}UO9>#C-MR|ZA0rPO42{a9fxnS{toQiU4_GL@nGR5ab+&<Iokm74d`4Zy=<T7T+{2n0C?CuGf?(I)?noMT%TuaD$M?T~Zf=~tub
7r7dsp6n6&5v_`Znu8G(dyVkY_>fxFk~yXt$Zx5@7TUHlz4=b^NFwhQRsU*_JC6!EMfVAU@@L{=!ICXfPs*^o{%j4@RcAzgR>0u}e$VzU9hlvT+z24Cz(I1c
fxi11nEGCP{qF~5+-1sq_Gf%s_W$R%GUb)HMO9Ojx<`L}l*fbSfWx$9+ULRA??CpJB?vQYM<X;PrSZw?b!stjfA2eMaj&sT75cNE!WXAR@%BeQQ?&;HuSYg+
R}i`(_^qj7I>iwT-M{Nc6oWCC9UA178-|h`>h`^ljUYVckuqCg3)w>&Z62;^cr-aY>@8}F*>Lrhvt_PWEK_Twz8i)n=b*Kmx=a`xG$CDXGQ#Nxp9#`ZF^FJV
RLt%xfOyS%k|2!}E?LyaJPeG-N-o)ZJx3RW&-{Ct(wKuIX2#9e;|&n@YH;?Gojo$V_pfDAD}&B5cgQH&1xx(8rKdVV(CnFXlPkgjuC+mDlpTmX7NGE-rHTOS
W<xNWgB|j-pV(;>8bX)eL{VVG5)%H#ExL>%$ntiZ7l_nH)%R=l8+8QyJWkc(8$v+rk1K7q2SdTQM`CqKSs8KG%5o-wr66k(4PYa)fadn8W)baB{6~4n@%2+l
1pQ?;c+h5my<hpN!#hP`w0XYz1EVcG8ls%+|9L{VJ*l`RH3-~%V_rtg{Mh@&@_kdVAsD7hyqUPHk@!}2p;OKUd%eDQCiSUd;@XQGn^B@C7K@SER_PPHs&eYa
suY})tlpJc2g19IZ<P9!2V7&%YBrMxBBt^5>L&z1txvL_OUV~)vDFHPR)X;{D>sZHB?gj{AYE;C#Pj;^3F@-$#Qgig@<^i$&P5k{$#Y7Awf+Lpg6!dFN8dg)
?u)Icll#S<J7G|^(rLC?7w;cTzJJe`ipKLDv%9NyAhU8fe)Xn5Ea~Q%HazSw?rV`7lN1Qw<#isaV?pTERO}h8^@j{o^jA%HC5-IG_YTgO!pHPPg!oN6Jm(6d
7Ok<vNL{a?^MD%+{GR0hyKay76p!2PKKF!*&9USg@{SN>Fum9i=?GHe&!Y~DnJ_I?p<v3h#CCpuIJm9R+?({LoK6p+@efP$YKi%*-jL=cizlYpg1_Y|IY1|9
P=PH`3sfaf%@a?C;N&rvA4WIzaLO~)UE-Q4q+So6WJz+u(8$F123sFA3V3G5*PEbX>=Db#k~mz?ll%;pvqkoOMtPY)e+YYpHagAt;y;hY@YuH@cxf6uo-CA!
tfSrMLx#MN!;z>gByWiA#|~fFBdih7Bv57W-r;{aYbVhvWcrg4lwPS+*`~W=@0r<BL3v}$(4R2!;&6p($m`M)G2$L9lUzuVbHuqD!vFS;NuiM5VoU9+83rk)
WqeC)F|q%lk;7ITG_Td&UjF_65p~}2T>Xz1k5safBxPlkUCExu-rL*Wd+)vXjO<xawq!REk|q+7gi1-G5*d;5^Zn=Z`|JL7?|t0Y>vhg~p8L4N9M7sx4qN@x
gmm!;F`Rb$g6N!99lbk(-jRC7QwD>HPIrT$z!Cq1m>M-RT@fJ`!1U^(4OAcBUM5sZL1eY$n~8r24A)2Ow)_f^B~DDAqh0{F#7ooi&cm4G``VSWKO6(0rKdAK
`Qe14aR-Z%9@uW_b{Du@Mdp;n@$dH%v1}f5`~g)Nh#F3Kwq4CZc9E>n5uZW~jvw7?cdZDAqP|7m9|(eWxuj(>eI(wVc|g8EeG92|_ekQTld(Qg!fl!qhnq{j
MY*>f;2mab%BO6DqVDSRd7RM@GZU$x)E<J`PpW0l*#uBdiN7bxPXSBt4h>g+CUpB!hH?)3f@evS`)Eu!NXWJW|IRjGneT7FC6jn0uk34^d3Y1g{5reGiM&xT
HO!QveFvoSG{39o<Kd;tvckcB1w&b`&3%W=uyS={pzClh9<*19E`;a8O?AO)lO+L*I+PfFQ4SC7E1rVd%8(K^mOL7C5dp<lEod`yVESlj{<RAM_g{4?B;N7C
zblS33xmEee42CzAKVerbd9FzXb@UiR<tO71!J6ScG8jF5+C}tT6d-#(WA8f@q?EaB7GXSy_ut7{e!zTk0S$nid1bYr2J8spc)|VaTTfJ`On9KVzJ3d(Z+Yj
030rvX+&~``0kzC_jDl*@;VQ1ZGK9})FGY~;s<_kQLnaq=aPu(q80|jlpAm+r&$d3h{e*6U{d}^j`*dY@%GtjKG;q?pF42Y6|wVI_J@3Q$7$`P$xWe4peZ|h
XC}uVw{-7^IeT6J&*@rv(vB2VO~)D2IbDUI*EEIFFMDi-T>H9F>w>r{@3Y*8vSD&nBu9EE1vw1&%#K_j;A!c&*O<-)#F7pkS+EU6Bj>+EB$b|sZII#LuBw3j
hRcuSK^y$4BFPb13q!W?QtjlCM2LIH8&GGDqcx3w)FsCai&0(2`op7kxt?NA&%X#bpJjgYoGTt76}k%IPp%<*_~OrJ2C;aqe?xkWzZi_hs(*SJ!ZA@v7FGVz
1|eh7@5om&F}7%3;8c}`8=sP82@ckn@+$E@`lTKUU#BB=S#wZ4lz8&x85`8neKs!Zy8r|Jy=JF6^>O5+>UYyUyW5bbaofB8v*l++`C88h*Cs78rF?Xu%u?6x
kSPnjpQo3noc%FBbgZlBLn!K)fAoKtNB~d%IpSq2UktyNW>tP+hL3WW?#9_>V1#C!``?Nm%4?&)s-9_u;Ewa@NZN-`<-F?|^}-f}^P3LL-7)y=aNWQ0m@ZCO
NgOY$iGlB9wNE~^{y0c4;I1*Rj8fa{Cco5UAZ8tF)=id$xfu1bQ}U8<byiXIHg!YYWPmm+yDSn1OPpOp2%v~GU{?t=MQSi(6{n3XaHYER+K3!tXXhw7M=l`#
wc+VH=`!@&UcG*|#2N-y_8cMi3q@Q1x(bD)5G;pP*lW+`q1Mfvq~0q7U&s#~eA>Vb29Lc~TZ=^~W*a-Z$Eg{H<ZSvhJp_<-G%+mqWg%lLA;Ymo3zHk)AL;SM
!GL4+wwHQ1m~6Wa93x7BPmIrjleY|!^UKjud(sUDp8cu6%|XEYiVX`(sXwwMWxn4JbOPV!?>A07wE&6VjZ*?~BIrIU==O3Z3g4T96rY-fV>7>tN@>asFFV_)
_-X^NKZ^00pQST?-0-;nNKgS%uW0GknSBwgcIX=+AOJu8$(T<D8~x|sGR&IJe~kj*<vZ%4eIgHK3gJ9AzLcZ#hJW@Zdm~<)h{|hg_knHUjUyDTIjG34t*L1X
#m$*#*Q+mt<BIo<rDl>KjIwD&*B<4;QK365Z&+hdIwsDw`6B^xS+b;`Htpa|^<Y8vWgt#P3%}lZlK^#R10_OpIHvm7oH_(j5M{J^->!8SL)X3&g1)-oio<Q*
D{;F%^NEca_PRJ<L^EBy>&s&l>4PuNXJct^+(O>*M4Vd<ugv`HfMQ!-)v>}vWN3=!9@q6lRJP2ce+7o9ojd56L#d7NKmKy|M9SFKb9bNGJ?@N%dF3c~I!4`I
S?zh_3YYRzok=dXu=eEW`sfe`8Vy0)Q+gJN6IVOTZ)%6oQn`mzTt;Ag!@9ArLJD+^W@2R-4){BC>WeFF2qGuym4`M>pvw0BO@_D?_!*<Tw~obNHBik`o<9PX
?M}9j>P?XucI7Q8ts`ERSzLCGK8FtDpQ%z$eIRXM(l+u!8|&)qN3>;=&>Z;G?H^k#{F*++1XcQB+fcCo)n0wv%uN@HF7gA}Lh()EPCu+M-OWBykqy5XAI+%S
88Ev1qlLRI6$Yu_nBA`I@`WUAv*R`afiIVi8Z;+@nqU3r!o@J`^hi?DHbmoHa7w<bp(SR<x3)V<jc};Eeqp1*8FB(+v&zdEIA$k*+fdpF!&9kTYDLlT&ZYWS
#jXrr6=yqUIz6=Jd^uC|zxVd#4(#k}F~I(;>?PxZNU%B4eI)g_z){*)8BU3g2$Zh=`Qu?cOxEw;A`T2hlVJF}P)%E~jJag_vzTJji9LR3&>D#;S|yTh;TQ?K
(^lT(1PdvVA?u(3gjwASNoh~U!G-wka8q|2jER)CR?EN%1&dtUW-F{U1n1tTb_J8n>4yq0J<;wWlgRQx2?OFTw<<2{p{f0$fVLXoQcSH%NoNbOCpB_u!Y(-7
pLWJFH5gkAVYBBB1mN6xLcdT%Ad3E3)r^f<qx#FuZsP+!sNWf>GC%BxVFu;rM$A#@|HQ~tbj}R|ad|aFd<9^f6=V#Oe1KAsQ+_MrnxNv>{mS^>7lKb8KL`l-
fd?_Su*uIf$eAcspxF^SH&2k?yH$!C?*^3(1Ovdylhhx0*99AYJj@ByPS9hja9}X;`%kBZ4qS!lw>(fDYA330?1_5^9S<66*r0JH?vBhOGxR-@Hy{o-1F2<U
B@0^`wDPmHZ-s?`<HxTXMF+#7{_3;o%eTHDUkmD-B=^VlOF#E%B{+hi>Gj%cW@QB3y;fKF#0|A$HV><(E+X<ni)N=~2s$Prrxk_`5gl{!)E|=&2n13O4YCpt
phs5RpYMb4;!pc;EEL1z$6e2xH^U&<RJ?!mofoLcuei_46Tna+`e1e>4aeW=CC^3$!>l%OK{iwc)qG>NT^$bKOY479o$Q9+p{23|RG}!3Jk!t;9SI4ABl7o2
Tu^g&vvhMk04?Xs^S^w{1)Z%lsXs*ma%q<Y`=5j$F;&G#fifP-r&KlrX#K$%J91vh+ytT1uA#<#e)w+E-}T1d1DvF`qfUSP5p~YH+gY^^C;Eklcs^yrsc0d}
p)(fb)ikr>=2gIt$zLfxnV|ndBcF0B5%e2*;frSiaOs+SxPD(WD*QeLY%XQNfB&?GTBIigPq&ZmnM^}c=*+$M-UW#8@!VY3^2T9#>*;{zYlv!3S$;0!07CSo
c7M-c1SNeHF1=v{hHLvHO`05GIFjcmZhRMB0ughmR9U!qk~&9yHWGgvZ~ZMD%0R)RinJ02d6=D$aQG||g^QL0?2~I2;7~o#uV`n1$gbKFsa}12Zfa?F3Gzhr
Tm(r;upT%RJSAhI<1uz{X^)Os-frKUTeT#<4;`-l3AV;m&~U80C2U;5z4O68onm}ZE=DbIlOX|3@?9wwy+zm)EL=wr%fs<F;<DcNDaiV?=lspDPx1H{QH5)S
1wvmb@5`RH#O?A3<@sHmXk1E8VtJN^WO{Zbrj9`DRk`unFZdFy0;LZdwe{dq1Vw?sqcBuND{bbx1)@e}kdMXN9fQ}3D|rUuz~^b9E772iBS&XWtO;g;#V6aF
zw{R5UXRX_Ju-v}J*(MDun<gTkB*a{%>*G`%On47321rt=brtQgvT4~y8q(^<9$u|*%JW*IpyU5eM5Mbs%lLcx`T<^kvQk6BPs`Fv~-pTF!?O_qvCHYeBbWV
8S2%5i#dCPO-2%gPpTIlSFeO)gc>O;l@G9YZ9QeI5aYG48kwDP(M#H56FwRZ`#l~3N=3oAqUYkP*lLReM~@5VzZzobU9-8MxeH3RDck4I1|#hCX41X`yLq^H
@AM(*C>+?icS5H}2gQA&GVxmuF!|o#|B5jkH2RaLekNVS-}yS`_8=}KyDfI}x{5>Q=rKi!;SOB+<ihawu08}T{9Mf^V{ozi{`{a!9m>3TBSeLQu<u>N9Q_YF
cy=k>*<a9xgRiE4T^6*0C3C#7zMBaykw5Abq0)y-<(_Zrnn_@+zt{7J&KPazeIe`}hZoXIe#N`FP$ffu{<^XtvPku9TMYHFth}mNIB^y7{r+w*X5`Qq`t_Uq
vuMmnRP}i~Dj`>>DC^rUA21SPzJ)zdMg8pcIuf7rI3E6Rz~RkqZaaigoE`H(duMHT{7HWlmA-TtpEH0RlgGIw?;J441rKDDE5lBlGOFudI?g=r@b6LaM{J9p
ru*lUXwH%fb6_igcdz)`QECr7xGZ3Oa?AiFg2u{Ay17tSyJNz0LmdZqIBoiZ?eXMjWv?hnFlOltuQ$2*g6c=<!g84&EJ__3mrQEFCrsvk>b?siH4du3KiUoD
7j{IioF&i}vLHFPCmspOMUD=edPtei%~^5E18YaqV3kw^T=ni442(J;aYL_b`>rn7=)Gs{hJ106bj12`ss((`O|bQyu)#*|xgg1T7rc;p8s(X9`=5W0KX!fp
Z{RBY_`Y*ajMySHV)O@Bur4^--z2;!&;uTEXoQx>qV<g5(8+Lfs9ox9A-d885%w)3#WUf^Z}qAhx|$7NK}VbQ2XbiBq2_k>k41B<___n7KJsR|Tg%71a0~Lc
I!H_*C15$cCtnG%V=*4qWQI^*HCWY?u>o;9$>N^#PMEc;Q#aUm5g9+4-QRTk;SFQUf`bJgxTb%|w@vt9<5+-6=U-oh>uxQuyfr}O$KWqFws!R-aB*|AQwJ5d
Dl3EH?7{c@B`<{{8y-;pv}d;P1>ZXw*(4K2kj7c3ope-$s+Rm0SvfDP*3=5uT+m0!#Ql5Ec+~N8WU&3Ks{wkhI|wD0o1%}bhIl|+8Q*!GL`Wsg(5v(G)RD>4
_>j6hYdUR)2Tc8KqgS=T(n8fuFDnD0_J+P!hX{Bt9d$~CMhNxnW5zA?-XJM<50&bdgfnBrUs?@4uy+w&>vgi?@6x|^st-<J=2v_ek#Y_vWe51}I9y?r+Y`{d
zMHq@;`cIA^)S8g&2{Y`HRvqT-0pny#&f-GmLsLg81?y{N&nmibzzc1pN+Y}A9Ny?qud0i2BMWkp^mZ}Er}{RG8kP9G=FkT6Ao=3GVKP`(d%Yy^kvxx7nB0|
7tO-ZZ*X3s+)e?{d}x2AH@G70k3jdSkt~qEeLl-;CXT1?@kO@V3H+x`Wog@$kiaFDN4uqq@}83-DV~NXTslS~m7#<ojgy%S{6=tPHCR1!N)k_AW>p35Xd>hG
25r8mF-D-?=pqq~Ol!A?!UddQP-eR&$}Wd@^?j{DCxsBzC_vWD8i+pyZ$px)j>Ajx$UoWL_2_s|uARq89p@~c-O_$+k6sDuBa{Im_^x$qC7xIouUGriTb_r2
X3fe_Il>BL8`QbHTjwETqA;#1Z-eWNhp*I5iQ^;TO)Gz+9^TY8tO~67AZJTP_pQ-+<O{VD?bO-A`&#Sounj|eV!i+5>Q6J=ClmO~bnQGO|FlK>WvU>)=1xlD
gdDn<8m%*%Mc`36TqQNB3@_!%uvG?5Bu%Kt`+x)QzEl(4P}6|TKRdIlUD^;I4kwCz9f;m?o1Z&xJ>dFa=kHnjbS#oRiF(+fh&@w>8Be)t!n>bFW<gRH)dEcH
kKpv5PHB_xW=CE(LDj~Gr!QV;KrT6>Wtu_}gXiT7o1~=ilwSDBD3KnH?Nxgz`Cb{W{ZD%$Vq|fd*Lz>;PfxJYw>&rDF-Fsm8*4lUG00<)a&C+?fV-9WL87JI
{wgPY<nIarwz5w1<1cpeh;+#Bdy*ZDt-mL^*JvZql=O8EWexVbUnzaQsRBmN;iaB7K|I>-AF;4<Mx*#w&6daE@ZWJvO^{JU1<7?O{WWFynIw;y9dd^d$JY~w
|9E1&te1;(ZkIC;@DEcE2cWI@QmJsf3W`4NdnwK8j<Mj$>-yt1xO=I8(tl1Kj#sSTiLZu%;h3h*PfJ^*SSk_p+pmGdcGSY>nH>fwinfW0127#-_DR0P4@H6<
)O&8KL+WCrUZ+7g&Ql-jZ6LFOMfmg4AGJ|fi7<$Ireh4VbY>&cHEo<N>7`Q6QUrC%YFja{BHsVV)O^01SLAah%^fXiD2~Z^`GPhc1?*1pHkM%^o$=fQQVm#D
`P@EdML-l+plykx9ge(mVyyj>1h1;A!h#Qdu%urbB0d(3Y>n5FNo)F0_xFBxGujBRg!<R(rqpn#`OJ<X$H`q@y{wbZ6%C4LF**P4-Trt=P=R8B9sB>567@G3
LCvP(clZ==D`|W?NZAl;dwIMSNU~tO(miWw5|8D>i)Ke9!r+;;b^J-H8)8QuWz_k_!y=?(D?X$WR&Jux?QWrXR6DJBd&~zb`+hjg(+0se_`u;CeL5(-FC%AX
tAS$!wLHokI&j*5l`Qd61Zs^VRo$H&Kr|Pd<{;t%_K{rAwSrxqr+;Q<H>QT~MbrC+#=T(vaqOn~#P0Zxeb7|A;ELnN6kBN4qhMW{TlX}FfR+wNj&@2{P){_H
bGRts>j{$4f?Yl8O0@R8By+)baB|gcCqGml(-S_t*9vM6Q>ax{FTnF^wfUhFdibHac{+OD9+qb6!}M}-7<wk`p`2@rH`f&}F233A1@_^Rl62X)ZO=Z-|2hqK
8JuEyE`%XU`+n#LI#)Q7<b1nuE)uRR&x7_+>Y_?d`=Dv66!xcc)ROR7!z+ur+AqNgX%X*af3BIJfJSY5&vpp3nNz2ED`cUNa5lwjMF*dA8sr+QjWAC*$GBsu
h%NtXNo%@OkWJc<(Wlac!nnlY#++n`3b(OWwi-fC`r*e4ttd1K&UkGn8{oR;)2h)513VHF*)aJef;)Ry_Plo5?SYy#&q`nDLt2S*TR~hAWGye&lPoEab-hKw
B8nSIr?18>O)20~<Cp&G1Rw12;mmKZS4AzI%%X#X9+WkeOt!5>akr3ZOTN|*O$5EgyfPEy+UAQg73)E<!H?nF%Lr^wMmKOK3qe~%?j2d45+b6AZM|WJe4A2J
Isr@Mv0r_#elrM8-+ou!BI5)bW4gPilmbFKh1>rzC7^wjRGH#8JxZ*He?8hTh3`>wz0+gLD1JSbY<)x+l>OiODesA5PTh>sM~)2_ON?V=Yt`{O>B4miPET0U
C41eah{NBRETQxl;wU(7U#hLIf^Q*_hu5DABW+cHY{AU{>UW(^3koZM;|tYCfj=}L30Qvgqcs7p@jTV<|HQ(4Ng*`NN(J1kk6)F%bw)*%bYAu)DFkY;6O)nH
V$Qp<aIcmV-m=vZi#$_AX$1F8gBW8Jr{7$#+-nR41_lxR4&c7<!Jg$2UC>jy{TOG{#5_qk-YqEN&*;X1%f{j$6pkg#H7KKK=Uc(QV{Z84WjeC`)*sJwuhBl^
5yTBCqko_6qH)nWgOHKU2jZA$^<@rQT+#e)nnouLW68@DT(l?t^Y6wNDYq{5yW_;psg5-TK~#F`UbUQ;$2t4Kin3erFmU){_G!rmcfvwx9V>&OAFj$DPVa#)
FR#C)HZTGxf|s?9Il#i(RR0+p8Ma;y_1$T=#jh=y;Oo5>7<WH$+NLZI#bn+t(Let8KWs;~>TU~6F5K+)ZgfH!-I@FgHzL8rQD6E-*9g2x(h_$g%Rz7#zMs#d
h=189MrU@{-@_5T(^kv+XqKjWA9lb7U-y2>U41SEHTyZu$XgZ=5B_<2#!&^X8Kt8l-!H<FtM>-K<4Ii3%%Qwh){B8$;y>dee`N{}U(#Xs)Wb4CNGj{TF+v6}
?BRPuz*7ou@*!^nELGL((DehwGN)(`vZ`ZK_$xv5)h?II+$vm2&P9F{_kd!W3_e|^Qjk9$0<v#km8@Q9p{49%af5_0hMwQwdvHY=g8Hr``{#sUPibH%_(%q8
G)jM~cXKegV!2R2K@rx?W7iGZIPp8y{(So$4;<jE`4Xb8fYntcr;NvXczHP@ufX0Gp2=IM1dP=1Hf5i1S>j=MQE!sb_g_R!><7Mx5IvkwRbSy8wFH01v2(ip
ywJ=~y7!b(6FLIa2~xY)cW?Wb`RyA9uodp+?!B*si@Bv)bo_>RElOv2NKzU%iJf_@)(xTP#2a?O#{jHKQG5ftIwY<9{iPwz3opJtcD~&ufsU$Yl=8)D5Qr@d
`f8*J-;b#R%0s!(S`82<Ez(DJ_}aR5g&4-ur}Xw$YUBDZ7eX&Rkie~U?nb91E_Wt&{!3Iuc>mXiKa(a<6{_hA{CyO^#X9co&ryeEV_+lO(et1UEIYFpWsdUl
?b$@8qj)bSb>KRk1jyyz(pBW0!m<Ka$?T8`EOPHYZj9wcv6D;Ty}ufGG5F%)khMHi_wva&-n4^L1|x>$_#k6Q7x#$D9{rao2nKArSUOnfV0b7Vhq9wK3R_Ka
q;64Ms)q+>Hi)Hbc6-i+H18TF2{+Wa=B8;h`{H?AgE4cGB&vdrKcA>q2J;2ZJpm>H2<kESAZwKXi)vfd>X<ZcH2kb{(PG54<u8#Jd<9{cHFxaf?z|Woq}_Fj
(S}y<Ap=<j2Ppl1q}}Ij1?E%oH+x)nbE4{^dF(}X=$Lr;h=^+AN6Uk+k`qR_NW&2LsPg~+_b&Q2k8VGe0Y+Y&`OA={4;>?U7ky@9aK5W}sMRS0)n>+>^+Yuc
YArVK#HnL?XjOHn&l*0lnVmI<jX^Q?LfLA>25SptBKca<c<y=7tN*|*FL`em`Dt?^W%wAg%fDDy5!&Ay%<3V1xa2m)0}V6@oo}6%6@{E_Yp$YM;BJnGIBt>X
fJL3vt?2Mc*hO_XOU0(*qvYbt=0+PFP<l3azt9M0lHOAspWMxD?}kna)7{?eRi32N=!GP~=L5U~y$E`KRwb-I21j&FHul75fUWb7*UyPuTxW4@GGesGouvK9
?F<B|oMFU)P*>E8Y3e=SP{R|xhGm+kt~gU`^<F>90Bp658DC}05F%nOsMF>Gg2XR}l=t2Ed-%bzP@^O){oc3vzxRJMKj`Z#iKpYZhG~9tk|loc$V5GA)qqf{
TdRFM0TT~`zDsm#!JK?qLE7CPiuR+A9Z9@!<E7@|_(#pqbNRe=@|!(~o_Cy^%~8Y(ug0k9TV+T{i<zlDO~k2>$tQL^JdjZH#EoaSpU|b<?LY42hsoS1%G0kb
@hVk4`^49)Xwd98ROB^AX46hd>vt~Dh7G@BbO?o32j7`j8qRRi93(AS;KFVC0@`GnT`stO{LiViFenb@{c52#!TDOU{O3J!IB1}3Ji+|{6n6uAAGr^M>z`i2
ws$(H<Hv8-{dtJW@5U60A(tQ^8*O$cHxu_B>a90lwuH~4Xj4;u2e>=s8%T-S;ezb*<U7)5;7r2uA=b(m6&I_8&FJjmH}<Ob&Y}Z0RDX1w*|~}l%Z8YkCxKW?
dMdidM-7Y=#~X!Dgn^})P<i%PHUxv>beDewp+6)wviMU9+&ar0MUQx3mdmdzm{ScF!D4d6i@M-`B)s}H&Khz{ZCBY8QgL$MMT2T}6L=rE*L!><7Vp_4G@1j0
FgMgil|=4<+l}lW__$0_Z_oAj5Ys4bAJcD54v2xlPZd#7&Oj6-UcO&;DI5YBrV@c+&PaY(c=fSd0-V2<7ME}*;!Nk4Adz2TI3Y0VF<=^xN2V`Xw{m5n8SLnD
Z^{pSCi8>$-bFz6OI_Z%e`fHYpkKZh;Q$Wn`-wfN+;HXRe4}M%i@=Kh>gr}~wEu2X+VEuqse<mOcef28Ph0=+36TTd1WS}ivG4MAk*1Z=ay}09zI<`cDHLz-
{+|E+#1sQtsaF$MZSh?Cpw!(iNj$&8Z(uTEg`g^`kHyc7(axbcS$IVi&s?tl(oZ{rZ_>Ah+sS#+RJ|{TH&hGdvo*Z_y@_D*`>;_ds)$op`3Go7wV)XCEj;zQ
Ajr2VFMJvI2PKh*xnqJNj;6hf*|AN+{aaS4KX>&rlA$HrJ5RvA&JyLfK6+reBy{1hqbW2r>x#^Od*DyZY@n<9RoHoV)vP(5LK=1WlcXO;sIc*9A_}m@clWUR
_XBLW73%co#8)NIB_gqElNUOK=(!N<GZ<3+`z|q80ki#5)O)FA@j@^$YwdFu+C~1g>}|EfGIgqM?;(2>81!u!9y<$<?C1K+6lQ4r>R`mFst>np>&Gh}YhX_C
^(F0VMOX!{pYkT*hShgcknQn-+1IhNR1*fcabY~`ZH43R{so8f$?OsMmD)w_w=62%RtJ<vLU8Fre6)ABD*~j?p2*CW#@>Jx>e!Ju$VaP9GPi3%Ohl6IvZ(?p
civ8C5e(q}gRZ2?&J%BFRX$GrB;bu%a4g#?0vsfC!#*AqhSZXh)3-@8bO!Z`)WZeu7&|O(X3FFB@t3@t^2yk<9<@X_E`cB}-ivljHZadCB3%C~iWCi9UX=sP
|M_>YrR4hHML7tW7%Qs#2;k{pd0Ca1E;LIQCEs?1VZoGE^&E3K$W+ano?4l~HAzT0vd$1CM$Tj9!-}}lbeHDvQW%QP_(<|SXT}uAidy780}M`IzCpYhgc3^~
M@2p#Ts~*pXL5MgPoG(7bxj&!=H7U%Ut<cIJcK%wcE?L2QE1--7E2@*nN>L}Iiu@*e{|Mv&MAlP`9f7pK$VY}&fIMmv=~h2y^`<+H}B23e_~e9tAB3CzO9XY
#QBQ`V`dO6jQsf?zBu+<V46i)3&C_>YZZT<27#2=?%bL<!loCdxRyfE-kf1N?QMcfQj?MNyynPKk$8BNa31eY#8+H7VuQ_&?$7cpe(+lq;!fQ%z<_au&XsX3
5a(wt$9?3)a}SM))8mFP`|#`yAtD&`#fRgnPlv#pSLfyVI7fW+o?>t^mjJ6{a?7Z&4@l*+vkQl9F`jdP^R=ZL7Edzm&E3a^fOf|PReEDQu&v$8q9TV4zR4^D
Njp6Fu;6QYR1Y$jj`QP8GQQvCBvO)4g;5f3z;vH6exB-``^RSr`TUKOc4W3tDEe&tomdD%hpYpag2GVuh4@TgjwURfgEK$+$>ZLn*(;7&V|1<xJZ#FbL)Qlq
m5o>ah#}%A(OLCJT<pw69tTr+Y6Omxu!my*PFzvJtpHf1ZXGAu-6nENzkKT(icY=<J1;8~F!xqA<LewNI>@hb2j7nZ>&^6=z4D%@RhTy?&M?Dhw$GBU0dTY3
uvtRg97n>IeY;fr(PuFE^sbH*-gPjYmpCec<9GaSWIy%=?{NjKD7G*tsH*PUx9W||kbvh|<U;t{qCLV==MB=6`zZ+t`nVsWz5hI!0d)6GT)Xu|4f(pP77^>(
Xl$%{{94EYkEmGpwJi{k&K!Jh;)5r`c$Kbjkon+N>8##zkOtf`UglKXv%y=XfD-$&Vo0El41WGm9+fk<1s}7!;EJkM`fY2o-F2^;|J~6T0~%Fq43$=BEd5?4
o?;7rp4+am%!(-NB=dRDA&pPn)F#r&u88Oj_@ME{7_uETXH*(^urzb`$B}otebwpbA8|QH<Zpg|HMg4!GeSdJ1Pu=qeLputBkhB&(#XxNU48V=<lHX$s|uoL
q)fSj2Dl{JRL-%c_n%JBijCQ6`I<rXGkLSoiW9Ph_zw3Fo1>nQC9KfE94f9)m{0Yp;a~%gkDZG#=zrYk)vyf0m~?D>VV*oVDB3AER@5;_b})x6*%`|dDjqb)
wcv1)EJp0R9=Kl{+n<aOh7!5|$@9f-7<IFhpcJ=5ef)uCjn}e>$iLQfomB%BA@7gOS4-`3(Z6mjD`D)bIZ1su!4;1kYqJ7w`Cy3E;P;z}D4a7PYgmaY2ZQvn
2NSXW82%JbKRKa`sh^uCwQ6>+W6n9Tt;`%AeHTubriS2R+*+4BQymDOYj{4}_`{65J4^J0Cw_S=)t=jV2*RfOeHwUSNrGT=gVY?IWUFb{-JMX?u+la7C=`wa
$^^?1XB6#c8#}9EfmuV*`NRX__>4Y&;iynBSHuTaF*o9|9#j8<vOl`7`=_!RhT&|w7}=DE8C1;Pb?@OfN3!ikNn58QlnZ4e<NgF;R{8Jmf38l*DE$$+&>IfT
g~{j289pGUxt+CSnSrX4ZcJz6V(`Sx?|sM7IA{~Nyt95N;dR~NC!v31;M;9?Xu+ojRrl<p#hUCeDeE(7rRsvcmp#`?6x5&-TyQHmp$VI`n*FO<{t))s$;gaU
ft$6d|FQT=?Av#%Kv_r)PO;Py3oQ;vIrv4;Gg}Q&rs~?fA=Wtdq?gyA+7Mr+`8CLu!{DH3UQkwl8GUh&((4WfLT_*&^(towOg_|{xXo&ZVn@TZK8;WWOP^La
9vFh?_P*Z^yZLx}v?SC<FAS^s4iGhsf)jI0M|Oq_`t39i)|faVW#xYA@8}rl#!4&J(9}bIwl>R8#s+5(^?g^bx5m{X9`Y0RE?E35AQ8H4h!8u6Z6bGTv@NL}
h_vuV>QKjV;Zq%GKg74Sv?mLNkG29h$^9_!;I)RkODH&-CUPCt*l_diRQL3_GO{w)+@<gN!dI?SFP78@MZDVjbNA1|Au9a$;X`SN4&5=Oaj?dsvqgf+S1X7;
c*xOK9fuy(-XR@cW%w7!u}xi8!D^r7R_>+in6;F*+A8qIBS~73`XEE_8mZWRc<u(>{Uz50+l0U<+`e>u%>kz7#%~%N+#p5#&HT?H1(X%7|B@cHM^t9*??y#q
Nd72%ax+X0VH5Ec_V>IY!|%f#?Q{bas~&9{7o<@2;mP0k{;>#_%GO(#^+Jk1RTka6BJu)qmxcxjFznqc95ca>B^80D3yiv`EUgOqHx!30X7SX^JbJKOzH__!
h%My2f6c#|JOi1S^kq?AYZUU;=Ut>Gz$Qg%!l)(;0&&zkWf?Ir`*rqd#BU7<u)002^}UIKv~;To854*fnpd57a|HuocKv&=9G<B_^2jFw=(V5MZ&i3ARqFhJ
_zOw!pFjJpP}&`=sUmmdybYkBA+g}Zd<HB<&HZgux4|1)&eD9w8%l?sMNLUoU~dM`oI->jWE4Ni>Be%S&@ojZB~cnX%~j{iyA*(Cp`w?<q9Fe0qOwO_0QWjx
orq<6iXzL;<Ad*ev0tomorO;g<VKv|`<X1jUYeXg+{J;iPKtMhWdulca{c>o+Yxr_J`w6AWZ?0iQH>DDMzdw^dTw<XcuSuAJNZ!oqlXn!xL3oGd8;_+f{HR$
mEOMOOwz}pL+>fKPI*G)Xm~N>|DMx1%Mg?hBo8T8&ouo{YG~9R`Fl-S3GcV<uaFH#!JT>f^cuSu^vIg0&kcuSW}o2^p)gT!5v%l0&g}LMnjgOMY%y?NNfwTs
^uh7QJl-)i14tKbIWC`e0#m!p<-y%~e4qo4198DPYstF!sM`^D7fUQwa#a5F?-`c!F(2!F!IHSdS~Z%65mHWl`z{+O-MKBPXSbVw{sZ-%uCd6rH?4P*=>+L>
9uh)T7&er1blPh?u)#3TW2ozl@<TbDGZM!U$Da`raoGv3f5YuoN8=#%hU0#MRsz-^ayh1k24G^}I&Io98`uu3Rad?6M?|xFny-~BE<68;w~M`ofQ9s*`OVR2
seY82bleR+GWEJq-UQIvANL>Sjf4BU-sc8_K6rlrqSpGn4P^AA^`06!fb(aMH1|d(ep6`Q$=K~(L$9v53-zg>^x`WUgKaz9)6VBop%p`<w|8tIpF4h3ThCnM
_eZO|>fZ#xSQK<;X~)XjqH?p1g`G4CzKgLlHWQAh=<YrzchVN?3sZc1J<afxHf{WJ3cz-!myOBE0SWQ<9{j0wh0MqW8r6G-5RR7$=X>b|g5V+%;gUZXS2>9!
`pltrCtdanyBRd<-1-jbTZ2I%X57t93aiY_r%HSha9#1fhxK(`gj@D#IQn?ua{HZ6X1ZR8woJaarQn2wN-Z7hgWQ1dJ8PryG*ldGrZmjj^*L&zoNJoC_#N36
BA1{6X`#*N%}O_Pd33b-9k)UFmE=9vELynUKC}6-#%|XKpDYY}+F^~PwzTfcZr-NRYfe`?VAkSA`u;x#n36OrQ`d5Y??WN3_?=z8T97PU=rceBiFLj&Q3WdN
$!^{HYy(mnIVl$!!vF5moz)&N!NXJEuBV1u;nRimV*`pdxc;N{T@kY~vMKi-Z{O`9zl0~eY>ON~bVkgY-`*eUTWh07cm0umO0)ugjv%WNxw^g?2?LfxJr~!U
F*ZA+mGMFf60e3o$v*Xg;k^}Vn@%UV6iZEhN;iV<zPrJUttL>H8}a$gXO8WxVOM1mdss_^N;REzz&~AOxoge(@UX!XkzIagAPZ}uBhv<L;1{N}5+B?QA^Y2!
<cO1P0f)^G>cQqsdGkk)aHQ)Auf!{cp;_FKisfMds*d>dI%w#DwxHtpuX$UvEH6gOmpg-K4?}T5fjs_Z&R;!s&=$*8-$Igw^l)<Slb4cAK1}InKJY}^gW_-1
p>6wJz5iUK7ne=OZMIi!;)+(FSwFq~lQ#+)$Mk9t8V}m2*T+-XZ81f0VvVuK3*Q)~SW`W${?lowZjI=1r4Vq^U&?Z*^}w6v5~9>dN9Z(YF~!9MKx%GWe^HTu
84gAjKPMLyM4a=Er?tT_FTHFkSvb~-OpaZdbA<hwONLEX{qWlRn4k=u1&*kMpX{Mg#lC|_Hfuazz?GTka%8R_Z2yW)<yzQ4sx02@o0>6%MZ&M#zUPm#2@xZ(
zJzzu0UA@^bigz^U?Iuv4u?K@tC}PJm}MN#9cgkyl>YmsrK>jBk?Ozv!P*P9aY0>0yL@4s-DTEQY7d{c4lTSayS-nkU8F0_5$f&IjL9C}VCdHK-?|YEQZ81X
Q?h1Qj+j#<5(~qG{=1lg4^A+ZOz&LQi^K7z`fd4KYuMyP`p`3Y;FDfv8nL@Crb+HcQ}ozCkurzqK7SZa8Q+XJsp^f3uRi3^?Q-;B(uNCPUpTnhh8QODd@$GM
7f$un0Fte(eoP`Q_`so6=i}*&-@lFY9ol2C58ux!$R|SHMV2F_+8Z9*Pqjz#o#3XT74zteJDRzxPY30MqLh5mZHp=l!yb-Z_lXkl_`n5=CGI#dSN-P6UA2SB
@S)>wldiZsc94A8)e3gbajWK`uDCsI#o|{T2r3g&7Pi1B6bmK3*zlAEX$eo#UvpphaTRx+d9Q_es_|nN^&LU_u`-zQNIbY|3m$*}?26pz<m&c<8@L@UpsaQ?
5^tR&4hAy>qx{j4e?2`B_~P&(rcfXpURQ)Htj(TaIQKL;&1@KiA2;Zh2YX;8&)nu(j1F?VMqX>aum{iJitFbS5iqHLxnGIL1-d$i*(^0(K(aSCop~k&{L2k9
)zcOjoNFA&_Hc!ofoK@{%Rpoo>Ytr`os20VhWF11arjL1dU3Gc1J2GbGLNo?KzXL7^3xkr%&(-}A)E1lz$5Vk&tyDt@l8Yq(Vty?e>ZUE{F(~?L@{!46CcFM
WnYyOc1DJXPvnJcO9W(e<wjK$;aMwZi$S;_yb|{}vAzy~e)s`mii*o1mm*#}5E+Vr#q#Hh^)6U`^h2OWCjzZCbos#y30Q69{zbcPhg81clIq`jP&h1jf~dj=
d(qPAvfGncOI3Jx22xS|PIY_PB@R#Czv2whAV5T#nf?*K51Jk4&yI*$Lgw#1{a}7YeDm_|Jh7RKcekdzoZP%ok$O(%OHVV1Y~CNyH?zmdfS(sNDEwjV%9^+I
&kw>bbkrihf{?E^Fma(d5T-$s(`K?RD7Ag-Mfm28l*Ql4uO)+^d&fv#O7aZm9P}>B#R@<yZ>ca&-vSu8S8@7d9IU&;bM{v|B0M9_dR{jG{{lKpd4~zuK5)X}
ux|h!KXi5f6=Z|v=0`*BZg!BikuPTv$VE4Yb$5f4D=1l>{Qi8$4trbg(pwui<JKIvSNeenygW!1DA1RWjIrJ|@&|%gd{MAnBy5bEp;3SO%;QkSUmW^IkQarV
48QI#I$@5g&)-4E2-oYKn*2^z;h<*};o$>Im^wSq#OcXEN&fQi&u=zhe|P_PaGgKGkE`8rHjM_OxLwi26+MI)cE0yAb%N*mPj%j8GxWZRR(NXg0E$;r$iwyQ
z-KtdGI4|#)c!PeoV)zbKTIY<Z{rI$x|4rc=uJU_7jze?@{lExTbINphu0-i#B@^T5G}MUy>6Zkmz=I>?izihU&~&(*5rqw!ZXnw$1Wn0(kW)_oh?rNeEj1a
n-`w&CGDq^vjMfFfbESEM+p3oXYc*04~GfCRuLvo9B}n1i5YanT7OWC=x8W{RLpj`KB%C7A^uLRiW}%kL`JLvH89T2Wfh>~h%+C9js&+l;e@?D+dZxS|9;kn
u;Wns1AmO!3t!3Fs|!-U@dG5>=2%(ym~eaB7yXLF<f9}m@M;Y<)E2UUIqSQ|shB|61YL@)8+ODzOfsadd0_LEjad4dn>du?&&a-D3p3sOSISb%k$bv*!?E%j
cx%Xc<?ndl@hs7>O_>>N_$`{9BApR1FI`di#21A%%94huCScGqG#AV*0fwK{BylT%d?L{5+!;BHEM#que=tP5&gOBQ-JJIrNu%D01zuK{x9nlJf*ZxFn^P+a
2oZhjKgOK}H4~m>w+a?ex-gBZeR>ZoHKxzc^nc6f%4=R_%hLl(!3bSkoe3JUQm9&(2ryx|@Uv#f0Q4GnIQ)rZFm}5`*GNJOt)f)g+fI7W7ix=kl*@usmURc)
tUO51y84mx1Yvd9p+NJRCZ0-)eA`yl$Biu|;yexoba^sJH_-{gq0C%Z>An~YIF^|iPinxoh$4set^z0(Yc`udu%qscNPr%VI|L|l6W;47!ZEgtH<!;8(cCT@
^+vXEr`~SLqE-S0<q2!YM>I&|9{zn|Arp-TDZX2h`p}u6<aT*zjj8wV6B{mZgQBxJCy!qjKX~$6naGSlVYct%!aD;vUGZSJGo%PzVIB44D+aJyDwE`<l|rSd
PmhASA*^Wwy!{6BAXD&E`Dmv+Xw4<Wgy_5xIrMsIUzRuwrY}nC%Bi8hR>`MFRuxURLQLvk?dtc^wrlev17!Y6`zLi-6suP8@q<#kaIZ7)WhEb&xO#psyP`OP
TE{!$=afMDNr`x2%@n$s?_JByDe%v>@XF4nGKON*g6}e&MWVF5QmT|05+o<ah3n`rd*t8c==0+E#&OOzNRR_JT^_O(E?D7M6YnHPJr`p1FVxN|X@E&2M|x05
9$9>kwW9{@@wtCTRYj8*R2p-2M<(p>n1m}WzC{ba&E$F4dJ^!J_0QX14HodGGy33^cN*mi$^QAj9I)?1-CIunUA{i&#z?R2i?T9pav5<kP@A3=V$M^=g`0np
I30u$M!eE$_E{X!Z+bn%B_%OmUDmA_z=RX?`{kG4o`YD-SrV0P6R<kT&e3ydVpfLzV~?mcWQv7zTTHD$ruE*4^q&j3|JFJjq*Z}?@`XmB%j$S7kh$3-L_lqm
*?2<1`Tumf8pp;iCvJ$1&$&%Yh59%{wZu%4Z;UDzjVDwplAxY`Ja#-o6CFLXlc#1h@bK!QOeC)fNY;ugP4x(%=<o4WykP=%*1+%UePXa*`B3qqUkM@|MNEEr
oRB0^ccL9n#NK}~5>JK*U^wm6-A1GigVx98Z4DwIFvyT)GW$aAo0QN!8!ga~eA?77V8tE=rq#zCsVLxmzkWB+8pQJB0$W=IP|{5-yjWI->IgkWUa2u8ivBGU
o%BMG`_dnRY7bP6sIG6iN8;S9Q5XLuC8$=(FYfeaBILn>pJb2~jHABQn;i2*Og?>XOR)>icb}SD`k{%K?0W5cMIJb|m2R$^u7_vYNv}5=ZNL@l?d_c7jNxjx
m-M`?*xA1srqP~=^&)weE5kuhk?-W!98N@2PwYYfzZq2Gjo&y{sX^jQtoi6}p8m+3tZQo4gM(?7kdlBMisy#aii^E)M{7_#@K80PM$9suN}aJF<E?X|NeNxF
;dwoa3ZSsD@NW-~!KPdHmb#KBOu|j3k562~tHJR6y|q5b%d{7HM`;F#c%Sa}z6J)8hlBoS2-u(X_;lcx(^wGoez#XB7{g1Z+}V^aAjQe5u<v|$&vVnMR96?m
D}&D__?&jRQ-w>P#1w?%E;sJ#B*2qkC3WojFb?r|Z_}L5g%y)QOV8yzd}nhV@~a)eNJg5+!xLrDHQwvsdo=@#C1Z&n`R&m=`X-FSzz*6Cr-bqz*`eD<Wix6I
A3mwM_#SL9fz9>ai9;Tah?dI|t?zcgJmb=Q4}B4gBENs?tFVA$8qcknK;{2EM<HAq76Qd)?Yq<#DcJw~{rAuQ!I-(*A7iPSj`ZsA;e(ML*!gK%N@b>ogNg!f
^6xd^mze&-&esMv6~+JT=S@S<o)-VVj}6fxko@I|QzTOVX>&cG4?!_W*05He9V92G+Rc6#qvD$)-JXq6xE&&<suK(Xfr8$<xY-Szesd#KvQZ$TEgJt+W{*G#
Mgj8GRE&L1*?OID0e4EgndNyxz&k*3x~(W01OETw2UDd%XVFno``#Dnn}ndh)rt7!u{<!GV1=%jyW6909pETw7N|JLg)CYRzM9?s-JLY{iI!U%y3;S~3q(24
{-EaBa<U$7-0Jca$+JZg;lM#vj{qDX<lmz_mjTm`l!JH7gK$WBT0o@F1|zxSth9&iASAXkKs_mrGo>c$qx04{enHeogW3qx3`-Y8t|>t)tf%?+<YB0e?!A0n
gbS^2_RJA^s)IgUA?t>349xo<i#6uRq2_MbkG5G|Fj$fe@O}~m&vn@?QeR)xH|1(RtW-js<DaN>qd3IqT<ZvWl8UmYGE$7E3?RJxSHHAf9m**??tiI`G5g+e
`rH{0WS1#Cl3l5V9eYg>l`<=~Ntji{gm&}xk|NKUU^_f9919juVuKgWgM^*?%9wE6b7W^f5Bl>C$F3jc#m%q&Q^7j2h*}ZTVEQHnHO_klFOrKOA(Bh4m}Q4)
KTDU5=H2!7ka$gX<}4_U^@xt#Fhv2a#iXUJHahNXPA|CE!T<9`9U)=`JRqZ9cKpZ*+L!q)XZ8KTz|U>K6sw1SQtRYRp0-H2dD(V|#SS+#&z^p=C=0PavL8(C
L*f4U>aQ?TSEPuMm+TP|Me~CpHj9ZA^rb!Fts7HE=ilqwLc8NkD>=HfFH9e!30=0;XS}h^c&8|2zadJL)ovTl>O(#>;eBj|5MC#STYnQU#mp&{rI|$=81Bh-
R$q|Dmf!tKU#<&yMkC(;d5;JTHPocTu9@NO=6rTikRY;_gEicZ*#7hH2_qjFA1yf)XM9Vf)d0ASzCIym)_}dHONe=7G=5JRRrEf+1TzXjTJ<|d=%DuF;)^sv
(YcmFnft2nQ!q~JRSw4`mGF{Wt7C}oo;g^)?gGxUEw(Key>K{DaE5Wz54RF@M<h0UV4QpG>ZKn}m{%Bn`?}u=QXixP6FlsZH9g1?{M!-)my@l<rFO_a&z-E2
8jlj2zdJ_)0`Vm?it*V+Q=GR>tZLkFM(jhAGWBl;2s1ySxFNOM--0fx)O5vQZt$7HKYABDo|xeOO{#;XjjusZf2pFrsBBj4kSrM6LpaNFQZVt#B=ka&IfnDD
p5^)$jm2$Hdo?=X8$S_^wwMhjq&eI&NAsYekh)2jv_+<QTfKs$J<bF+<P~qIz~w->&!v+dn4%LdeKO*U8O{FD>5feBKTRAO{A`c;dxd|p&TByBA<^awva9gU
5^r}9EJ5&r<jdAh3D{@reC}zUE}XBYtp}Y8!m&!mA6fS;aq@6#b%UQ1QdkHXon2v=Q;Dd2ci#&gzewk5$~f_6C0j@Ms27SV$G2xAOyKc2^77fM>aaA8RC-{=
i?uW|l8rmI*jtizC`a25u_tqw_8e8kVt>eu$pLrNA2SU%f4ti>f(()$X1F2GZYe!R$qJ%2^Qo^$lED_3_MTJN7n*;}_ESuFV}I`7DQQ<1@HmMoV|QLE8=a<p
?)J|P(~n%T1XX;$z&aGX&jGEEKO7cQjmH%_^=YwI4G1n>OKLB)#aqE7clQ0s5GJ0)el}Ztn_?dPEaZU$>`(L#&4{CxBAfW`WDtxmdo|_y`GWl9v3<SQV=&U*
u;avf9{XynjGyfGXK&4WOTVKCP(0?qN#tsW;45KG!QYIqsTAA!XTcw43%X(qqTYD%=E~1RN=Kw=Pj;66^#*;mQpQI)FVvJh|Gd)|hxsJuYVBxW5Nc;tZCBjU
KE7Az?6f8FeQfm-Z@NRID%Ys>mLH0SeBbsYT!zQMU%_m1D-`L9=dpY@2U+P^tsW9J2$+ovW?$JokDLp;SF<lhJT-3mrs(48=YOO$kB#wvOr3XFPyhe_g(N9y
kqXh?d)K|a_uhN&z4zXGsjP;mC^V2t8X8DKl88_w8EKc_>-*2={rmr1=UnHU$K(FE-LC7L=kv^5oNBCh0496s3r6d5@F0`AHb*G{9^>Ptf}|-}=N^4GT@nKF
VJ>3l+Hf2&3{9^TbppNN*$R=@{-{!2rhI(zzkQ@fd6z8QqMYz*kT+N5YA{m!+M+oq!yt6&`Ao+7EKHs`y!*P>0@3>REG;q7Fnjr#RPmW1PFOlWCOYDa-`PHu
)pts8?TvTkiHUfabWw*~I%<vCD<=x=)%Ea-(U2mHFxOq>n4V|q#X|?1F_rPYFf+KfsrMp_a6Lb*hlHGrDk#yCj`zTH<+(R~K_bv?`8~;3m<i@F`;GFv5UAIz
M>HE2pyd>+%J6{zxV_BI7xA}7+Ra>Vtt7%+s9uWJjJLpfo|GH7OW;g7$#|jlV3arV9uLtA#nZpDR!?(cFmg)9n#|i56YJL2J2eRiTatUx*>8l<#)knBS=RU|
E&1vxr9JXr*Ol@AHO2UWUAy;OK3G`P3}MW%Mz!+rBXxQgIIs8lPnOoBUQeIZAlVV4#A40jqro7xf1}eB;SAF7r-=Jw4)Je{8>aih;C6vUQI^~u1_?NI@kk`-
e0Hu5U30)=E-jIo+GJEDrM{k34a1Y;YGeYBBJpbIL@QZp2*$blzZUuWBKx!2dGR;)a8qcaJaQxoN5eIQB&SosD;L$W8=QmuSAQf&Bm*&Sa%4z#NeYbtBhp*(
-cT<Kuw`#_hU!4ndEwL2=+*i_ITYsv{psoqn|((3dvkNI>Yy`Zb#7E=MS6nf3d?nwpN{CFGYwSjaX`#-(dx;HaHt;nM0JST3Dw#=%0z$Mz=GHPp)x)Y8Pit&
<K_r4(mL*!GJ(i=TxPOD>WDVRj?uv}Q_wj7(4e1p2c7uKkI@BTkd6LX*RJG(lUpY$i@II$x>avu+{7N9;#Ne{kFs#FkB0l~kvw>QH2=+W?IvETHE0=a1R?X=
3CVetFl;)8T$mI!fy?|<nP!YHpi}UdXd)arHE%|(!fY{Ny-{oB6O1Ooy&uyR6JT#};>k%pE4+O9mD|e97w5t~>A6@!aOiT02USN30&h?3f9<7(-@%_WnTmWt
m2oSDnb#b*u8bIb3U$NhD3xzk?=rysw#o9xK0C-(R`XkJ+v9N0nc&w~18`~HRiiE<3VPJERrZ3WpeqRZvlQe3EBmPi#o~OllV9`a6!C!SD`bnSdEmq8(w%;G
O@x{mEL@NCLaxn&vA;(x5Tq$rpT{T-OSwCL{j(i$WigFzl{p!bqz6UFWZc1<%q#wkKMTx^e6@7vf?<<7vCpn9621y79naqzpzxT#t_8m}zVbybvBugW_`6SD
g|G}BN$kixSh0ex$T8&$2b7RVd^a~_Oa^E2Yq^atL|`r>ecCrG6Z}EmCr}oPy_(bfV<{FGn;txRr&Iy_#nLY4BoYZe>b_BZ*ajgTe@r&E((sgSgP&fCkdut(
xck<F;Pve`3;m!Y6g8r@b_N4*$D&F0ho?PGgebF=lKUX`t-=kxF9I-gBpDL_8VLQj`}za(oKP{jNc)BsD8Kn#^zd6(RPnpWUb50fBdJg3s~STvX>ND8eu}_-
Aq8pX1p^ezn+%McvqK$k;INRMC&(i5^}f3XgDs#i?x2+mF4lUnZ5=cKKYeS+%$ylSG8SB@stqyKHfuk`;Dfx!33~e3viMSv7N0=lj|Qe#;^u%LB<%lsCc()C
Cgnb?!_vVBQ801-5^RK&r%I}36SjD;ajeT#*d8bMzMdfAbpVeLi|Qo`2e^0I^PW1Rj{~meJ3BVU=#>AQKX^U_ThmKyhL*nY7rGxmpzQ{?zgJ%B{?NtClPWO*
OkP-9jS(jeh(@ZMYhUxMA2|3w-w$PQ2EU$a@gJCi&4MR*?Nk8XYu%`Cp(E7US?N6$3|V*`)TAY2a`wM{cdo?#Oy!FPt8p3QWvK#0(uF^<boa)$t6jBBgMPTA
abM_OnKO<A<__JTDMEy>NMM<;H9YUx=9LEML&r<JGNS%G?#jpoJRNhypz2Ysd1o7tXLii8k!d05%e}3iHU!=)C^6eNu8bc&syfpLY|!}Z^fS-!B(#1=_MBUE
z^Nx~k;M-!G2+#EK<%0;Buur!gX+}a!0W`W;%E&jt+P$yY;GVf?WxFqV~b?9>v5A5ez+$U7kT7_K4u=?`fc1}fU%qVc$^0H@#Rm@T3oCiluuBLJu*Cm)i%p_
seWvDu1&(eGpve`@cS~vIVKS2?NFt72ar3zV7c4Fil+T_WZtay@GuJcv(L{R={HomW6oIMvuK6(zCH<5XwE6bDhlIU-@`|b`z&yOPsgePQzV9Hj<YurY2$qB
*VR2P;;0S$)p+-qBn<kG<XX3AAwii!R$kQzU8^IxN?(*QKx}ySa0xH6^XYG!TM9z!6!$yYQ)*CN8#3P)>4bEtttM{~3-q6nXzNNd#ze-@)h!7i+4!y*6~W)T
{Y9wh2>i`pt#p6(uod>iX0wEoNMP~Dpx@Qf5Og(J@E`M2fq?YiM*YL4u*l{pIm)ksM+5W8x0mGcQ>MZ^;GZHGdU_{MZWH>Qs9Q^km%x3@#D_~FRWbP`^b|Xp
E(Z8Fuk}qEz(#F?o<~9hagVEN^PY+0l#$!J*{5fqr9d*P*%pRHZ7I<W!kjLoaqUyNU;@gx^YZSx0^l&1A>!84Mu+V1vCt)gkGMKpER8DTBx_bbaU%zaA316*
7V3ljgTa>RZBux^o4$7QqckS#Srso63&QhJIjz8s3H(W~%&k-^BYMFklcYfzTSCN-y<7Ov+!Ge}+1L!SvwvMK>=@y87)xnym@#;|CZrudsDgI(UYM<_Ji0aS
4kyz};Br3q_nb_^d&{hA*roD@ZI+EtOb-imHFrj0`E)Q+9L=50tpS#|PAh??T2SBG(<uBx9?FOPhJJLYVnVpQmGgiK$h-C&iqV%r<SK=v_){(1Ha)&3^2!<%
F?!E(l^DRq&~9vYLK)OipB1iF*h6AZ=lfn>Lqs0XKN!d;i63!2@8YKwu)y{&|Mz(}RQA|$S-1<})GqO%G9pL(4*aIQ`uo5A-Z>5xv;QSmfPTkZ&3hU>XmoGw
Tzabtr?kWi?m9~NJ1gNbLC7h7e`l>XFO=ZLU;csYo)X;8I_I=%i(=)JZXaK}3K$>epXXuK!(SHlv#k$=vGnZZiLN0Y2=8zDA}ekJ@2ZD|%$o@yk1C{%Y>-6<
n_0e51T)T`eabUGu7XO<NOGrM7M%V;KA>$PjXTWyhibPxP|@NQk<G7)_(r3ZwwH=1ni|@}pev2k;YHS!emhvxl$q-&DS>nKt9k*E0F(#B`WO{B;rOWWxJkG<
I0GCLo31IL;b@G)y-`ij?EPrS;Uxfv5^Z1NMP)Eiqs6B_8fzbZ*nWFt1C8hPXYxoqp)HbX_wtE5*k<Gl(@ezR6t1apM_w2q-9j>>X%x6&60WqlW`p;?WMm4j
NW#!zBlEqj0nV2U+nL9uKr280+M8v046UhjB)S@-`d`>Nl_(pKbU(^jHkU`I!RSDk5(l_<SID2~OJGf@S=vh_3My+9t(O<g;B;+b-%UYf5IqTan!s&<8fweC
(F~^%q`t;VTC0n~#E@LRqoNRyDIiFYBBFlRMxS35f!ZPR&wn2XU@s(RU#lp<F6H{gYx9~=9$xBLYmi4u-rS-6#RdoxwW*vJWX1Mi_L}%D1q2F`jNN!gjfGb6
`9KM7csfy7*A^bb0Lj*xsH`-&%QXVu<{yPzcZTL?J2&M0^1naHNAP#n-i#F+9ds`iJonp?fry+!gaBJ8ZoHSJKkv*BF83pG!pi2Td|#yZ)X@|z+znZ;w9PS{
lw3EXrUZtYOzvJLj@YtmdEF!{3w6h!8ZlY{9EuWD)Z&)|%fj%#6=i)~JoeMmphg@Wj^a~`ISw%YdfW4AuO^{yX;$B!VS{Gu9}m+#l8DRJy=))s2L6AVTT-Ly
(74RG-Aj7_Ur*39ob6`Elyv*mAMzR~IUgVSqgNWL;&f-i`$>@$Wj7=%W(-?Cqn^mK2H+0ua@BU{!Z%gD(g6|~d}n?r+E{xUp}cN4ECtwb(!qLsVL%p9{$$+J
>k4>NpO}^TT?2tyrT43x<)Eo4S@pt(6t+^+4u^hN!P`%5SGQUNXBb#^?=v&wTHrW^{9_F~r<o9cU1W%;IWJR-YB}(nW-?0g;>E#_p;^?c>R3N<L}G`82Fllx
jyN4wMeXqc*WoQmoO{XG<jhUTov)+al)mnS`*P2bZ95L9R;#sqEmgcbD8BioQWMVh=~C}Liy~z2emd=C0R;S+Kc!8`$Hd5cgBCpIC`#^nS1rN<Ew}Hv|8mW7
Z1nAqGNEY9X6<cDAR_qgzaff$1M(1+)sEhO!xnv<bfRUs0+`n#JF@FQhPtD3WgQl>u-t6xZat=sjd;fST~}e;E}9CbjZ?+@+|1|`B|LB-<8!&|!2o9^^`58!
c9=)myw=**!Mnf@KdpL6Kt=V^WBqk9UYb3blAsaAb7SAAz2rj36?~HSLR<!_5p5<^*Dav&vEY#;`$-VRKg+m7s)qzuQX*|C6|9&F?+fVH$Ib!WpfWN;9DaL1
?=UYNwsehG=fnlEog2}zd0QQtFJ9PBFRaSBH+)OeQ#Jq<=T-7gl;XH`>T8QW_gM(_(;olZtcuukeV4EP6+rE|*)%3bL)agm5zsg+2x(6i<ylTKTwUy@?et-R
|4efIYJmvGXy4qXF|vVg*Zz=$wDzbIc)N#9lNz5DD46%(GzWw38FHOcb5tqxhMGJv#+X;(euFM?{5!T2{&!ss4@k;bYu3~-zjON~(X<}KEmQ8_?6g3PhS=yA
D-pOeiB-OvGsgJ5%<QD|f8S@V+}({zS9C>CsgojwYbt~fRngsW;)I3Lo`cC;Y|wtDvF@3r56v9bY6Ce{Tyhz+zWB@@du|T6oeznJ&{*-3+_D=OK3B$*ykSN6
=&#-8ba%XdTyxR=v?6lNX`_!5>Pb%PcO`w85hgAkDK5Bbk8>+3#F{>Vh)H62!<Lkek=_U`|Fj&qG>{Bx$V4N(cFa)gax|#xooUF-#Ne8uAhJl`0y6G^s0r~)
c-E+ys~wz&+sd)8>u1amPm?b3(#Zni&tCmXku!&AhJEw!vKd5E$iE0yDT4mhNfyU;b<B1MwROP?WC?#Vi8&g;8}i;HLCgsKe%qmj8*-SoHc%1WQwQ&ot!9Z<
ZzS`Djg*NSB6+@z`JA&VPU%QlS63z>T5{?3;#YT&nNt?fWdy;ma}QB)pfCcN+JB1PQ^nHh?LOKkV|e~v$cFlz7HBGYs7hz#@vzZ+hBrqEonz{|*Y!BiAbuwE
ue=_-WWS9?zgNatZM&?_cU5S-D6;BtM+ua&+V_n$Rq=4@LPL&;9b#$5gq|dMLX$)D(%6|0eCTJJ+83V))yK_WDcG!`eg5SUQh_wEJ?TuN^dk7dd+nt6`KCB^
JCmB`dLT;d2h+EbZIG&QC)K{u3<W+qJX`^`NZ4yd7joSmq%M6vURj0^oNedYtPH^EJv$K*in_SK?)8@TQ2<7{q;uu<1(5co_@lSGJpPJFDJN%N0sYOOb?w{I
;9*yIP;fm3wTrJpJ=m4N^^7-u1r}(J-0OwR3|NS2-us^5f`$r}<hK_LP~l%f9Dhd`@}<T1HqHnmJ<b15Uy}p;Xjv14yTT!!Y(Z{fZjM6@#Bo+HoI$KU=diOk
68WNUSCd;z!5ONb_^w9<n@J^!ZzWUUm7){n{wxtm-RG{<zfr-`;jP_AZ)6~JN~J=>mj;JVpS;M$;{ooUT8fQshPY_KP3n0<5cUr^qkk1UL+;zxn^WIIkX^Mv
%}!$s-lR6u+P&(ClCnDLP2++0m&>GCSqkBzsC}<GMja=Yk84eSv<Gpvbjwmq05m2U({g`_;PaQ_<NI$LLtv$#*U8ZVnM59O#&gM7RZN_wh|&Ya<}8DYdlvS)
L@+Wc=|kB*A-+>F81?JA_K|}|cx?RsbB)}8>t}T{-<p5bYz5{mexgIUW>C#-yO&vHfcsy%$oCwVM+}+S{h|jch>Q1c+O!G;i^&3`7nKCm7u8+UY5j0STkY|}
qz%%B=1gt4P4HY#a<bA{7F{Xtk2-%ALYa9%+I+nl2A4T<mb+>|_grJBwAKtp*}I#|0zB}^O?@NIk^n9?(|#>CdAvPN{4bSq5f>$A#@QzQVSDJRq+g*u$~rtP
9SHT&@Pwp$do2w3+ZLm&co{q!J~dixR!9zfT(K0Ri1SB(kVO#Y<f-B^>zc#5*#7p_{Cu!3rh~>`@3b0$`W{1b0ks%>GP}5f;`Kpp_~O>nh+AMfl}UB|L^ZCE
YyEqgk&e!XA4}+5LUFt9dbUitEyllBHC%SlM@VFyx0y2^L^~<Z44MXjtGQI}wyr+zg(R61NykIxn|{%^l^CdNa@JB4_}cbk_}H`Nc+9Bzmzwt0!j${w$J#|N
*gPdSM1wxcG~$2BKh?)knuRV8mt;_-i_VV{_#&_Ca&@DH4ah#LIL&jVVTC-~kikqJqlNkbZZDE>(tu<qX+{E8E*?2lX#t=+&Jw)LErt2-Q_ig6<H$3U<<@<j
3!M*frca+lpjOb<(CXX-h~Z2uYw3zB*Ib3xzc7RGK0{KRyd{`qxV}+%O2eITmw#ne7oxTH@(JUzV5!fv$z?GEcV?yErBpctpJoerU6zFA)q{qHN4=3jKzFuE
M})ikJhxf%#~D&N62*h*@R&>f!^kp-3AZX8$81};^Tua>{g@2$n4n@yw=Cplr?Fj)%)yXD<?lJm5J=|=FYEnLMd=r+glBhT;Ifk>u3WE!m?IKy-;?7J9(CB}
iHRkAQX`@VM$bXgpE1z6A`A~&n$Evv)56~0t(+ABhNwG}2Aj}eoJ_f`xbVpgvArv$uKuSH=r_tn)+34T4EJH49xJ?-<*UA+APAMg`h%5w1Yp7Fre_;>0q5%9
g|TylAn+rN%5hRdXvqGn>)tYhNgV67Wfdc^F}=%LyRMDtS3xDCrG_wK)he7mWQ3p8sukgkoLEleN<H<#3E};I$^NH>VDP!^;Li+q<TW4Uw<wIm6Z`%0*Rlv+
P;`Vx#Lob{+48pRR-Q;J3DM-J=mQZ|bkf|oF|2;^(f<<H1AP`ge)ZRYN};Vv+dh8;TebY?7<NZH^RIrMo9Z|#@bOirP&`D%%3u8?_;AI|FkX^0GaMz2&cF7y
4b4Q&tK3}<_#zqlpf=zTyh1KY^DeMpiKV@VD_;WTo<%);-3mCPaHL|HN(xoSL^N{4GT=P^K$C_%0V4S_oWcHzIM8ue!=~OB|9T(g-#W{UwhQY_%7x;vebKTu
HpK;_$Bs-#PbuTqi`}n(%=M5oZ<n&smyBD_l=!;!o#8$lH=rGN7DuiAJv&+@jnK2dv)>!!VzK(37)1~#UVG_1*GRd92x6M@^X9CG_dlFWPpgcq94m8@8F74#
BXelBV}<&ur{UTMC2`3A`L6GV4h~(KF#j|{$Zvkt*u^k)cq(pb_t1JEs!x(z@q{=;4!8e)C2tI)EI#FD`O4_J-dD`|WLb{sxLd{9&t>3Q$>$CEVF~67XFkL>
34lGW@omoC0{q$?J0o(3;KR}%1r#57gN@RMskq(%4`bt);;RIZs&mxbJ;WL<X7|q(j<{ksFSqs2MGLqn)X0<i_`$hQ__{tHCHmj`)K-ePW8-U&Rz^%NIGBB>
S<8c<-VvuGFChf#=U3eN?VTW(p3!`7OBiQ;?;lsYXoOP^3o={36d|7f*ea8b;Pu*;O=<$F*fSS6|5VicfBBti$*Qha8h}JE(?*YKAb$QN>FTZ|IQYN47r7(_
1>s9drP8h->aCX@wiJhux!>iMgEg?q>!XNgtU})vx+wuJ52SC@JWRJ_L6JSiQmE|_WG6GndO;3_Pc~nTd#J$KIZx<RlOaeWZ##crCG??A{2|Te7__q7?^ftY
g=UknzxCrxgzNKrwA=}RIJv)~;NN&uDIXlBG!({%k4jFJ+zq&7uv}P_bq>7A|DIMZrbGMZs~uAo0w;VMY#VMc!4Wp$AqQGxEDCDpj;Q<NU;F&eoDww%%)QK+
8nXviIz!h-DmUobZ7nv9U%{l|uGdh2Iy#5)m+WXn5c2-RH-EYW@CQ#`JgOCdFF_ku@;ePN&3ram+guTAUFXN$jl}WdKqkxM0(XoEXlx|QBp~belx9_j06x86
@b*$MhF)$DQ$qSE5^g5seSRT{y$zC8Di@_tz}>s4I4le?xBUm0znz7wkBaH@PihD%j>{GMqKX2I<6*VGav-LwR*^lSgagFC-tq0#z~mJN)2C6bNUr_*#md7O
WZAd9Ehl}U<{#f7rIvsamrr3D_h&)U*Z<nsDg_6cXPy6unu5pC#I(l16xG}goYyyfP+B79NG)K6n=`B}GM1)b(x)igTr)(C0oSDnDOIHJlm2r$-4L1!#$sM^
zKC=!x<A#Ri*@25tv7_Z<HGsFZ2Jf&Y$EIQPix7;Wbo^7BYy%Y??*DFwuwNUP2cn(RS3+OKUsdgPvFbnQ%mJL_IQ%b;W^uzf?4VV9`94k@V07a{<+)&&i{GV
IZSviH?e)moPz@)jmaKOf_nJW`s(gX^*gM+SP6^oG)H+uGxs&$Ae1<)Q^j7e!>n<;MdU>T<h?PjOpZ2!U1%SPz<dCDeHHZ_NkTBPskolOAPM#mw+^dzS$MLG
F@66|i>2EhvzZ+(m^dKG6+z(N?tFHo@BEzL&VSr#eAx?u)$wvgvQ9W^OS13Hh(1^w)wW#Z_`zUwepJxi1>!ZgUi|!1jAIfs+0N$5Xp4v)OLca|*n?C?cPeKv
3pwst*uxLDv`m?sx+a)e^x#ytwZyQv{`<Je2t@5tDTcmR#}Re$=y|&s<R7GnfB#Plv~~AJO4wAua(EwaT$>g|$?8lh!rcGYr*4&RDTg$Tpfe%Emfa?YK@Rri
(mz&ksjD(5laYd;{$-IeqinD%T$%eAX^4sa2LAiS1rSKWX&_-|imaXW1Y$C4T>Qj4n?hrX(34h?TnD6}Zd=&jMI;QN(dpCH1g_7p7-?5)sDQvCIh$a*0k~^=
%#=#`!9CDSynZ+WgB7pqE(eMs?#)B<pW$I>sbwUhKI;R%v3RfR9*&@XM%z*Gg9{lhivt}o3J87~<ElAYjLV6Q9CXoEc=TdkYDiE4cBGnZ1yc5?p+R)Yh&rMh
xJqhoYr%@FuL`z?_@pC2I`&Qs0rob+ijjKwY_Zk$&ipPE48{|5%r3#&PkL-rJ{4wN8;^VG;~|&(U^LD^A1v>@IZQXT@ck+I>wypaNWQ?i^xG&1y;Cxe{i(Ha
R>)&tpmzdp*sZfO7KFp4>6C7JiYn&sILR&(>h_b+BfZs+m(fEOPdm=yhoHvijA^R6xR$xn?l5ZrwQs%0`gs$vR`)ko;Wh9hN`m}cge88`og~?9BXHJIZb2D^
9zIUBEoqn};JtIa>aMRSd|t8t%e4swSA$NgtgIyBx9>hJD^ddWO&Y7hKY1A8JoC0pDH!>6ojTggT3C5fa#h{J6=|EZ_XE}Wk<ZM3N}Jdi*Ci_ji^>T(-P|w1
HDr$9(GO299O1*Pne8pbI1{waR4%w#0y$R>xyl{O!tt({#M%*mP<Op9KUwUC$l9RQ-8nyW7*L6QCknyRHwXQ<<&I$b)Ug|BYy*ji9`+3HWSl-eTCAIp0ij~9
fY3de5dO)@Z|xWfB^NrjM~(6@3y!~Y&I1_E?J9ZpNC#2o8zBw)UReF)l0emHg;%MUqZ1lS5Yc~Ke(6#O7CAl7a_Z}$&_(J&*j)qo)f`eNv`j|k{v83g0dpKZ
a*1QS>NJSB4D$A|OCpzFk^EVu8JsN@D;fKE@lkkV;Cm|<8e>knDd^<m6Qwg-a6}NiqnwFzN)51(PWyMz$pqdljE{}A2+tvUK59i@6Ar`a>x`?$@bnqm9bMPO
dCdm4tVu42jc92GEV#k%HM1t?QvsYaVSGX>?1wv&OV{r<MPa$%(iz^vCSXmeX(;}z1Cl^$1#?w5{ImW+C3r3t`_!9<{#F@-Z+R@7gI^avaiaXjM{_ic82r*6
wF1}dsT$f}7RV6UlQ2ZDj0fK0AKzRrf@=Mp_RG0^5K!in+#_I&lsgQ0hwolTeA1?&DS>nEZfga(em)4hMlQY?>f;!yKR@!ERvt8BU+0y#<Z*D2eC0r$B#6wH
))xY^FzmL-x;^2C6{atx42}w*QWO$8FBAZ>-2>V&`COpXU~8B!RRQf0=@zjGc2wQI&rwKte^l#J<g4ojID4DHq<cOB?<~{Te=j=W;W4cbFS9s6zYyR%KSem7
>1p~E*8+raJJ&=DutC7+q@Y24DIU`95=`|ZtoA<UP$L`{v_1azr6?E58265CGO|KkGI=Nat_Tvp&_o2ys6y}5cF*ftd-R+9ylk>ZACXl{*BEmH@Z<g=%A~Jc
u=Q+dJs)F$&;u;@L_R4(cvx_+@x^a)p3!->c047B_PqLV#nB4ff7=cP^9tg3@cjo}Jp~Z4pAMsR6a<Iu!n4;#7AQVNzdQci7}R@zOs7QgVk){~j`6-Dm>Y#k
Z)P}yL`&w^TE8_|_S>bc`*@)A3Fo%j3?;&KT((7XO$dGdcV>7@9xfyrqj7w0kW@XkE_qBCPmPQ^4)&;_u0(!>@ti!yULW6ma?2217SE_F^n}6wFR0s}M;^}$
`G)q(sz4#tN#UWa;{Wp7ICq(C;-w2*<d*u+?pFqvf~T~Pr!drJ$`Tt(l+jXQaI0*<9kD|X*WByG5zACHWY^h-;Tsik&*NJ0<@@YX9H|4AbE-Y49OyvVTBI~S
77Mk#v{d(046);|c)qvC6TVKV=PlnBKv_BN(*05=eA`2tPfZ?)U8yeV+PEUHaur^YsEWcf>4}x0QG2L3`#oXs^@q--g;v}%L(%TbyZaKoAabKB;Si-0!Zzt@
mDy~Ne?OYJqc9ldMkRcCO0H0re@53oU<9|cD_q^JB1kFdl(L@pLdUm`i^~ao@SouRt~B6-3&obF)jclaLnPCiKf96GNb33~yr&J5`iw6YXkBpNOK74@WH9D*
Y^5CCf}kza)ZX|Z0fz2d)T1``IQy&d2#Ksa{MSEcJ{z=!#bRoL3AG)@15WtbjS~FZoyxq3T?MaxRsYH2Fu=us(`7EY<xn0Pl<|(Y$EGc_!KGhjIPshH0R4|t
3=bT-GTUj4F{O(u{c#bX?!LgwotglTbS=w0Beo#65zVIh!wZL35duj&1(;<!eLVA$KYH33a<#6R;?~JeQQ~dvxUf}9#M5Yxq=13x07`2_86=M=X4=4)!N8h!
#SDq{AB8?hnIqKwRPAC+AVSng?%W;M!mXyDlx{Lx5WQheWly)mroqleL~|B`d;f3?=PRJzJDz=4#0N#%{UkGng^0_@zj<Fj51U=D)vHQeP|*J^4Z_xV_2j0I
Z?Qcl!q)2+-iBaBJ)Z86jSEz@kI@)2W<$e7$9y&~2}h-78g52$!yvnRkEESBu3x(MmwCS-LXu9-j>P-p^LX^`*HLz`(U;~*#YbYfkV82`(G>R$K8)N_cYuCs
dwiUaCQe?Neq}pn3BLL5W4_hCU@vYLi7ZM7=fyDErE*P#$mDq)kRW*D9TIzE17Pb{w8W*(H2j!YCr{bP!}F-a<k!Sh5yp8<Q$pqfoTMarIO7U%MUS=G-L?dz
6qcTAX7V5p`*+DnCk)@htVgE#9bnn_qVd$A681!K7E&hKVkzX;^6^W`uqj*Ntv~IEsN!Y2`|Nqhy2$VUojD4c_mWxF8`H4g%X&vY+a1Nkk$bJmZ1I42TaYp$
3M0;f{DV}(C^gJIYJWKmQ&O)ax&Hq5eb#Ea%AV<=N<;)i%+bc?;J{1r>s_ZkLF4o*=cbk!s8`y<hi#n^IJEz_E`tXG%@0rMovw!FP5X8IfOP!EhNaVqQW&h$
-I)<iLr0zdUOJg*q}ggja9#rRpT|Vp((wd~_P*ghi7@QT%B7sK@<H{4;-yFe=Wa-~d@(e<jk$V-FZ+oLF{Qq6qasHSe=pVl%~_~~#^t&%dqN!Wm4eH7CBq!6
Cum=3)ri6|EUz$WE(pi}thD*ZspIsQhe;kM!?EtiNW4|=hk4_qnqDz|oSlvzF4QnWuHRAVJ8%6EP1?>|<e-S(X-AKH$yvdivhGb3lOd`v&(&$?c!KfKwe^yd
24HOW<QLd)0OD8+0eV7il-<31>kGRW#ww%}_wZUHI6nO(<A-F}w7=kIBDMtG+N3JWZ&#F9Z`e`i8RCZaE$N@$0w@wE_GIBTz-`HmU2S7^#C^1JzhUBnqO0xK
Tq-=F-<^3ZdPoK2@1Iv5`>2M!&6Jf{-KNO+6+WKUuMeeg3e~`0J~&)IoWFK12x}JMeZ9Y|Fmpj`&jo_FT+4W##eK~ROH=fVidsg{J4Qlr%#sTxNklP$tR{F#
d*XxpA5Y}<C?p~OG_GE)Y&T=n2k*y`e_JY&=)a-DWh$JFcGk%Sw)h~NeLv~lM<0Sm?OUHI4fT=#cqlRcpc|sURM4@i`=EW+&Qnt|00pXK8eTi@xTM44x-O-T
zQQA<-K&<kFu(J&Ps9VWUV`RN0u*sD_T566w=P;u%hp)Wg@U6y`M{Ep8ER{5tuBWd<7>+vHcze?99Nxwu2o`zJ(tTaj&>QKc&`*$jiv@T$o@DNhlF7xYT=uq
t1FIGB~=%5x#8&C*GM8RJ4^`EykD;Z`22>4FWOsU_WN>C(+P7N$ckILmh23LFg-pu9z!(x?5Ccew?;hIt`<eT4)p(|^Bt~tK#W4@M}@C$5cqa+b@;q5I%xNI
vIGWzIOb=(_&o<GXE5Yz%Q}G7Xd*28s2fP4S)S^@u|T-NqsSOB8~oC~;Zmco3$|H<Q~#_KaP}o-#lc}y9REk`p-GspBvNd0>UJ(*;}@&=_9zI3^AF17O_cGA
CBtyxPyl8PM5;_o6~P)4lUL3pfQ%R~W-2ag#7%`Yig>z1Rj#Sq@qsNArLJ2&u~Nr~`JMTqJx++$wpO=Ut;0ltaHZ;7GpMmdqWw-BOhd;+>?Ay~UzRrZp|%zT
Z3RbkulwNQi>1uy+uRs@;B)_7h7BrCB+FFB8iI3M(!xi=7TX(P1N1!3AnxAmeU<5gV&Uf1&V4H2+Fd-tH7$z(26y6qQfqLgGH`t#*T+MDw?GqG2ONBNHN9}v
2*!(Nu95aiVPZHjlwzb9biWo4w;$EV;kQ?WC%Xgj=ks>+&;b@y_bm&#mMg+JlaW8`4&gpd<=wk=RT*MzZoy>fK^T2Ed8cW=FbZ_@?)crb2K|cLbkDm8*xlf2
aktTgO)Tk=<~z~&9dEX7X=Dm3>Gy$oexa!P#@O$&VvGe(B59>|J=Casf%8`)PCjy4%NEl?cJQ6FVlrcp{wC*0+uVbHzG6X(qfX#F-Fk@A-5ep=qt`ndjlr;L
a`sq?9TdqjU;flq#l(37Jra3;81J(!Ir`laCp$1_6~c!NyTzeP)J{-)GG6wF%mWMLv`_kEY!Udg`S+|xFiu^t8sRWfg5vFI4>}POyo$VRXs+mp#y3sxe%C6Z
P;NI($jKN?^)>gBN&{f_SvI`+mnM3S5`V96aKuczK3QI-BkZUzyE0o4>X`oE>j#~LLG#c%>wy*H|MFY@a6O;7morxN@0Zc+7~lZSg0@tm0zS8-X8bjd1P|v0
Z+cZN?9UwiWA3PnyLagF;{;v6ed9P&GqoXR$h^WOG%iC|=SbJ)su?u;{yOxk5b`3y(Q%)s9-^E!;~l7j!LnAyf9*Sgi*=cc%ZCW_-u$<4yuBY%3a?eP>!!ep
%f>#?&lhuh&4?@<ZSe6Lxuggiflu7un|~cN!Q`T{_Sp#sl!*R3ZMLP2&1(Nk`bm;-6}X-D{e%x*TlSf|ybZ<kvXP2!F9{s+Fer0smLHZ@(esvNZrJ(!<r=-%
S%fYxWc&zp#saaeBu8-qwA}gzZ<6KX0^3zCS+B<^3Bcn|7xa+wr@)Z(uNxk_1o<Blae@t-)`L`vSlnZX-wq+M!7~kRrV>|2+;+4VESuCrQ?XiNShNk8t5zdF
b*n&&_?6R+pd_C4(Q>apGzO2)=Hkt>sfa&V?O0G_g#DJqZzJy{<IMI7t#3dfj+0w2^D|jO_s%1;eVTUY`xcRAcgq7rGflDaL&mTo4YiFQWP~F{ox31o8rH}A
iEk^(<Aybrq`)O5u>ByiaOF6L6YW7sp|Prjx@cHu$qWQVp6!=qHzSN5CXOAdRmRNIGPg@LmZ145z9;6jIpnkL=;*2?AaYD5)15~b3nOn_g2=34@Rzzi&4Q3O
FKB5v-Y7th(JsNK%My*VZ)9nun=pQ#MYxeD7xx-jCJsL}L-7IO8g_4MC@7Riik9j?BjL2n_ZKAyH7Q|H&P+nk9Vwb}@(^hJ8LXgga6%7dooSqw3P!pPnO0mS
<Rpt&ki;P~tQnPmt4az4b<az$WAkFjD^{i#yW@?B8z+35U&+Dd#S{Ouaw}9xxU7pyn?Qapdtu<BIh1X^7dsR!Q1XQ-YDmqW;L}%}JttJKH@x?Dtfd+(u27p&
&MIN|=e_LMm~!~BxhkgL8G(S=r-HKUdbs4gZR6$GjO<$uR^bl{A+J~2LUH~aSUC^6s-IWDgF{@+H?CPC#&g>ufkFuf(@a{6Hx!^!SF@`ZV~#WX$DFP|5=VKD
A*r@*AJ&#0K5dF^K-pUtMvE^|I8&=|OsLTXQPnE7&%PStt-sdOtJWs?IzL|<rSArtm~%UOLSBG$^;pO2jzK(qZ(--mX!`%H2e0xxRQC<Onv-p&P?mvi{ne@W
*R4REJP^p9=!<q;Ce0sfCODk=nd$Jm5FC+JP%{j8hiQ$u?(at$Vg5FCl=o{Z9^_qXDxI(gyIs#Q-}H32h7A+KQU$Xsx&vk#4hZ7t4y$A)^moU-)R&yLcsCtq
&O+l3u|(eHZ#>1Im3roPEAtjio-FKRD3yb_(<@o=Wjn~69ZcyoFvGX8o2Ea{_~L}1#4o*v{2->%9;Zq$gu9b(sGhDG@;;s2BF>M*^!r<qs+D2r>D1miE3AS&
d89er*Y$CYzUdap8+%laQ=dvYE`?w&{pW#-h6wx=6+uVv$)i8Iqnn+r(6rP?vFZYNNc@;T<|v76O12B+QHB^`5*2<oCx8ck#&R2m!f}U8<@cL?)*xZgKD`}n
1eHEvo}hFq{LB`Ayu%GNmzbD!eqsT|(r7WcyF4}?kxu1ht03{`N>uSa6Z9<Pbso49h{a8&iA;Yb7@fY>U}Z0X3x|Y`{gp65wF~pimZ=tW&TijaeBgjXPt!Ub
mXq*xpX$M(4;m0iYZ9^eY6#t_ea7ZIrXcqh&S$u$14Z@M^5;lSL1gzNht3rPaOT)g*BDyh{S|KNcj{E2I)=wZYFfBX?iw_@D+uEpDHd_<ctp>iOeo<@!EuX=
RZ&ko@P;<=_P=QpJTBq?V3?Q;o%q9rEUQlNHRlXxws1k;MfGz!g#OAYqj=4kq>JO!-GxR{MtH-L8x`ztgTY}BmE{Csq%o#XNG;2PZ9z7Buc;ftEH}v~T2*nS
g}Ux}odGoaY6rHIL*RyF^WP85u>AJJ4ay5n&@|L!Ig?@yj*_|!r}{u-R$gIJ7moq`vB`6v2zhqQJ7`HWTpz2Z-@E&NiQ{j|3!7*!6Ik;7Nt1K+frqu2jpHC;
t}gTb*-vQ#L7hFL2P%zlB7VlKHCzh*k4!!m_?p8`zA{#Q$`N6^&jgO!8$<ay=i}H1x`>s|A}Ln(h2*+(>@$5$w0@wya;_&Fzv!|;EUx+D@u}~N$C!;#DJ}Cg
i&+a9?@v@Rzaa3xiZ-vwV-<Yy`WPo89e~KYfBk6Bd*U7Yo;neNztHj2U6#DajcVp={%<cD!tY#L@UoH|9waL24L+d73&sa*Dc%-n&V6KXgx&`uK{W%G!lBrl
`5?DUssYmj0~vQj3_(SHZp&r!8cM~dmbt97LH@M!gX`1_TvD)}dP3odjk^yXl^Tnp>Ba3X$1)dO*L%y@k<ANxWs-w-HoVv{JGs!{Z2-a2n}2oa(sAHLLaS`6
Ev%b<M7}7{Ma<*{^_g)=)V6s&?JSf7WtF6`o0cwtU#j8)^7UcBJxaYP>W(~#yS*3m9l^z}5@6*hjQQspa_hBiNNZ;lml9NjaS~P5)XQ`v9m$L1X+H|H!#tmk
9a6yMq~t{%N+YywmbX$baA9ftlij&!N2r<6c1}N)LaSDhKHE_RkYzCCzIk&V!<wloR|)6OH2XT29vy;I3e!igpK3v3V=2Ci;4PP5uMj!a>f=m1_2tfT2`r15
&iHn}!egOBT0PWKF#dYlK;wiic7pXh_Oc#^(9l&?*8pA6K7Y#<zNwCCqgm~?LvE-ql-@hrZUSrlwoLmzUdXIneEvYiAGK`#ZUd6~(5WdB@p#LPnOpVVq2&%J
*1wRxO=6A98<{N}IVNznSQW}@w1j<8L(K(U2h1iuq*3-)!4<V``HFELJh?IGaN9}*=C?!(imjape4nGGcHSRTZ2!iSUn`+nVEAR)Wpf<X`r%@f<$@uXq<kYE
eT=$Ni#&c~gU1n{Rwcje{a=1t4!_F%;c1KUN!GW+4e}7&k;uOyDUJ#XdtSEW6r5Up*2~(f42uiRv+je6$UlCXBxF_}szv8m14e+`BF6k-B<@HOt`ALq?+b2q
Gj)39aCB;2mL{@@#hHmPZyD!MIKONXIJQ!Z>YY0SM?1p78+|j5PSPI?`m3ftT(U6bV0GfAQ~=hRc>h)@1fuxPxcw)K0_bhNig+lMfa=h$E4LrIA*es#OcY%p
%JO?_?jI%ONYUf8HNHYPn7*UpC@6sS`;RoN>i#&DHu|wI+Z-o;nf<<h(gF!|`Xj-cy0DvR5>nVILtgT-Va0AFzU=l5p0y~%HPeAszKTRlQFN9U?fO9Rb(o{a
^IT{q5Z!zECjw1BOijYCgyBtX<i5XYMu@8jk>dCphSQT>rE&3Lp!VEis1L6}k?X3=IGYDn*+*HU^HlKAhs>Z~G#wV)#<7ReZ{eH!P1Rd+)lkoKxgxz_4|`Ag
YB!fe*c`SO)jpOA#ni3Q%fyjz`64+P{-Frs8m$>~4w;y_e|LcSp*VIOTwVx#@rTnSJF~_IM;vj9IV4JEfy!qqPZRCLVJobPZZ%Krq{NT}1)j&E?|!s>b%J&q
ue!Bl5lBx5hI%!Hqip}@37VudsG0Yz9t;e{zq&pJ7s`6{(Cp)5pG?4Og|fW4=rpiAoNT{u90=i{raQ02L-5|8{TP{04yKs2jI9pZqrc(@Kc`h7zW&|!g6new
v{=b$Sgai}|GT!Fn8+Jv{3pGO#7dDv>t1-&DF&>rUAg|uxmen&Ci7a9N7%Riam|aB7#-_KJf~NRf|u@WzXhwnT>kfc&lfES6rPa&rJ97uSAIW2jGa;LH*k*R
Z7?2fR7+fY>5t0Yn~Za_shH;edHwLA5M=5)kl$#Ghq_ui#VwVKU{uaZ!mT2-7*8G2+{wU7vgeFNdJ@92f4}Cs8;q8^$4OEGPIx~s`j(A02XkSs%KvmHqhwdq
uI*qZQu3@iqe_GEkMz`Z$BSU_sDCoB+-n8P;r^>+192GXF<Se1pae(d^U716SmG*k{Y=n}Kve#YUimW}1&g0S>-+{~kgE|DsIZTLo7wGu443k8(5hr%Co&z=
jicT?<e6~&RZYqFBMP~PneQ@ml)&{{Jl(u|(0~6MPH_zlI7Py$^$n99MKWHlRgb>-5D%J#ac`ox1vnH+P1e%lis{86&olRGU?nGP&`jt{u_vyaM0Ba3*}O8W
Gj<st&8o^$f@|?G@{_h>x(~AGt0i83vxV#3?V5TKX9D+-{faY*fDI9m+X?4Hd=t2QU@fZ-Y*KtfBI8N;eVJ^6&*n1Xl?F6#YRbb!@@GeDRU}O3W_E-ACLnc#
`@tc;OQ7v}FWO-k1iPlYYn}0qAeZ*OrS#Djc@+{rxAUAK_4ydN?uskkt7RIIp9sOCe)ye`m$7i=ovmu^iv@8xTkL1|6lhTXF**?&4cVMY|C=4wIAi0s^hVGR
?=GL1dd}jH{+VQX6(w)5WmGbjkoaSW{@6&Ij3qud+9p-a`GS}4sgn`UMJ(7UUH-ApA6=dYSM^8JA;=ZO--JkHRp`hlIJ$#l`*WSvSTeT%Z54-o2*hvACoNy{
Gaz|jMgQ4p7r3~pYtGzG!9s+ib$jXmt(Ua(E}l9QD>;uh<gcfK?ar}Q0fT5fm5U@H%5p`dO#H`h53;e@ZsklZl?68P2q%jl{*aw~pOdnZg~FZs(SPRRFydi+
DCX}4H%GQ`O4lUloHnxl^F<Vqf~5fmmO>HTcD7gexd+Z(a0*;ijYaE|Q9V^se+0cfCnRg;3Ay0?_RqhEKu0p#d~`k&vvu>IcK(`zX5&+N)XPA$m8PAbUkrkE
?~@#TNhh!kl@>OqCBk2m`aaW~BMvIuYB!)#gY7-b!EXu<5dO$DU9}Jn-yhqD7M=&7M(7z$$|nL{<+&&A>+=F>lp|kG{5hlrN>C^f`pJ(;fGw=78n<M-S(Q~H
P*xNlUiU2$silJV>d)DOCrkBwyJ;-i>N~<?bd$h|2@+ekAf&#yec#hF7?N{k9GNduaO*B(XBoE_Wc#0*d*=AzE<?VFDI<YfikTkve$7Wj#s06w<@zvOxGOs+
7=^ne#v2@T?(n*5a!%IM7w4p;=CvEbAZMI#eCdx9EZX8`CYqf=;k(wiLK2Qd`po>v@m!o3nTjh6B5?eLp!Cg|ScH-#NVst%;&Ix3+55)cctE{;%APU><X28^
=e9e;t%7R&cvmd=ehsM?>?3%H)!F0ePc>k6ev^LiVHljVyyN9evLF~?+grWlh`}S_3h{f>!SaAM@kn4I5{hp~2{Oc^U{+?Yx|KbI(z1NN#0KI#@gqG}GJj~_
U?^j|E)Qvw@uOUGp=enD=~2gH2J5B60m<$b5Kz=jI()qV+ClT%i}Ydm%pP!Kl;Z#9Qch!jl>=%tNI(9_)W^#=my7t6(r`n{kfiH#AR-fxb0zdVL=v@9Dmu;4
=2V_;Diw>0&5m>P5vlO#T+Hp^xqz{tYt{5r_RthnS8|#OfY1KO3~Lz^<VmGyy<*M>Y2Zp0O?ViXO|l|rTvXw4#Niidt3TQf^{z@?w+6AV`k7n00bucaZBZy*
iJAdd8|4^dtS#`p^f(#~Ar@yNJ?SXe<Q?oRH;(}K$HN`d_ll9Uu}fFwZ3n*3B=#g$zNi;Z7MnCPMp)+EpU*-vvB9a3c1J@Vj~KS>iV5?u_OL_hE2dnuhB}m!
cI4yJtIj842ANoFEz~$-V2AFZ^H%AeVK|f#SVolU4@iz)^EdTJ#l0KPr5`!MI6u{R-r5Eqn4(_pj5s5Q-|K{3a~9m?L@m#>#3JLF;qLI`G=#~|nG+K{%<!uT
z4NIgEZO=}Y5xs|)kaj+&5B5TuU(yb{5Tu>Hy^!VoVLfsn@mznS~19wkZ`b3h==52KS@Pp8-gZtOYc4B{$G9{_c{IJt4kqvN7+MGXM-?)Qe}HFm5}@HJtrre
ypduS%04OQjECpsYHv2><Hpf%RXJ4NXzShC7CvZ!4Pu40lM%6?_$1IVOO+3^=w**caaF{8_YWBrcY!U<SV&EH3s^XX&YBaraFdqE!X-!y22E}k{!JP~Q$w=5
g~S=%C+l;|_K71<IPz$db1|N0Q@1pEi{d$7Wme{(2of&w>Q#SHL|Jc$>J?uj>@{@lKVrj*6V`)Fq#|ml)|wK?<6whG!mDk~Pf_r2*+>7&ln8t)-JgmFW5AJO
wh$})Sx)!3Spbt6!QUzEbj>Q1KsvnFchOx940pG`GdP<glwAL7QX2=JY@Ugiek+O}T~D`XQxq_{(G;?HUK4p9?fNTewCJ#S5Y#(q3)^A8V$voaXfD{<&tH&7
%i7)=;}vnFgtKNnIY5OI^5=j0|KI}Idiv<5s12-0OscXLg`v5_Wp|O10U;LCYm)~u;BLyr^MqUgM7mLedUKpu);$t+n#%-dHvSnZtV@6_@zr5Xp<TI>_D7{q
jtA>+#SX!ja^U~0Xe&LSjqBplf(?uZ;3FGq+kHR^7d1(`#H?f>|7JtQ>5MAWDYGqZ81R8Xrti_+S3tt=vEM-_1mN?B`00&(>X2He@M%#Nfc_q;fA_x2p=z)4
dmeT>JiX2^&uGqylbm<i$;72WPi2x^dB+N)uU<SfbXG!e`mo*4SsA47U8#L^@(kYQgEpp10%srf`l|Hmp)qhOVx;vHs@#-lJh=Ekysu8t;jt?2oXZd94F4;a
zM&(z@>2?KZ<e=(f>|+8I+b$#kuqYAv@ouz902W2nc&l0XK<?boHy0dURZQ2wUicd<K_Zy^A_d*BkH`Px%~e(u8>reR3t<h84)1~m%Yi}d+)vX-a9i9%FdQb
QnFGgDiu;j5kjJj@bmrW^ZWaK&g<N-bDr1Zc|Fd3&iy6@Q`c;4XBd#K#vFNI|3z3a%#bb~F~;SLXUS<VDDfov@HYb)HAL`mFYote!$UPo-t4U?{CV7I?^LFV
n#L7R*)!z07wF&O`PUdi@qR|nyA?q>%b)Whzy%6&A2UA*u;bn#!l3JO>^Mi3UVkQm4wCdLa(2Fqcsn&G87y!P&V6+5)qRICBL36u20ahRbH?0$1gm0cBj>`|
TfF!_we9`HQwhzq`vX<hmC){%a7Dnw5D$YZB8@!+A(glAx$ChSrbazCr?<3l>d*=i=OKpwa=IHKt@WNl3nv=6OudqH@kd2{v}RTo65bxK&fI0j(#7b(-xK_p
l{3-Z+G5B3Q1-k9Mrly=Q;(1E8RB*_Z61BOCsf>Mjw@Wbh*O$W5#}#tadFnoV5I*HRR3ldYZD6LQN(>WWn+Mk@$;YehxtJ(A{|G1K@hH*^aC^#s-P&~J5Wv~
iz(*m<8vlQ@h@U|i~F|;Y@7a#hoq~5>e0yqD;C$WFunBnc_XmNXZh`F`<^d5glI$y>BIc%r$NFRUmVF83xDu18NYtk&1%$N#Dw8a;k_19{LZ(3BQ>dym`)oR
(TqKgTv%XXGBAU~iw*ryiz|3W@bgm3Z%tI3Im@O-EP^>+l`!QqayZ-Y>UU>_CHM?F*!*D&$q#;6UYiy8R~|He!Xg&0{wk7O_3?m@!nG%^`((k+EwP$lbrr9|
KdPKPAc;i_&maP232f4?-W%pogXu`~3RRf}%KbNIc=9uGf}Oc-y+;d!a{cbq*LZO5s|)Rfia0ifhi=)kyP>iDtcLO{YmA5HuglF><0ucs@5*r_bZ7A?UiOm)
x5nffu>udQKGc(OP>_S*(EP0eVLHUd>|B0190{h5MV`bY0~Fp-x;m)Jfav#S_0RH!K(f5q`MKsgKCKg3N9ahvHEZnQ*Y-TDe@J$cK9mMBLm|mqp5dTfSyYx)
t%rBz>7lcGy^wWm<hpcT1lE`dpM<<H$B~6UF9{kHpnm$>bB25kBuBPMlqp>Y!S+*^1Y%YkRyWHgGgU(Lg=x2c)4Is@-RlZ@I>_WW&DDP)1np98xpjroFm9Z?
dm>m2)-`GNMf($AX4FLUnkyGxQ-`)+Czzp_VD6(~U>44}EE;`3Wdxnr2QEU?Vo>`vuw3gThK<t<ZqCx$7#SAz{8bi#u9?_T;Tw|JjS{)os2>JhF=9JWYad)I
$=TPHsD%I`=TK55X*?1i4{K=iz?Hda6hASBjpr1WR%79{`Z!u(F&!_Tt(wM_8G-QAN2#Uyd`$K|T~yxlJBsErT4r*-xEzxkAWyCa=6&DS9eJ5yFdDk_?6L{W
k5&bEG^HSsdvd`?@ES^=57@9%8A9@Hh)mB06AC|bKU3&eMcs*8hv!AFqS~HPcH!j-D0Dfm6Hdvam}2)Wm7_Aq9<L4AF=wKoH;ZWFQV^&_l^kbS%}`-U@%NLA
0fKX^zp4tWfyqXjCr>~WPiJKU(p|1YIWAjfcTy5ts(rt{<#5C4@SR&ll-r_F-l8QBR7s#su9zM?&5N#@!ruh-Whji~b5JY2gj}-;npeNXkyA9l(E6GcQ)J>Z
W<)xuvS~?RHQ<4+$wgk-K`AI)5Mh|H_C%7*ms_{)N~2h1CA84(GQNh`lj(X{VBa)#k(iz)j;~c8ri?DarwWGMzZ(n)Or_$c(ojZ9guxBrDpjP-%O5!DxGT#2
D0ozxNe~+0KHup~$iTNZpNkI<Ve8%GrrJYhXg2B1nKLtB*oc(ha6%064>aWJ@=dULg!onDtQK~xsu}jnG9q^yos8wO&^Ob4SlKNG=N<bydRn&V<FEMT!p;Ya
5q>FQ?fqEz>8;D@7X*dY(dN59B=Ky1==l9pB1p;BZoCs?hHnM?Bh8Oo!V=5POWRpukQ%+q_w=?23dVaTd8A}9%Bu3FIx!1*C*mH-l3f7R)BP`Y+Ae}-RdrWY
?J}r5zE|~&`l9a@6(vos1XO*bXWjeQ@j!kbWlH>cjMK`KIz80`tDjQSHLmLrk-q;{^)xp=h$#o#F1moXWd5x&H#=BQ-8#F>EREUpcdx?o8L+50e@krs;D7wi
ts}d?O0>u49-f=*%wkBd)L_*}62{=8?oRXPIbaya;n7+bT+V)e>a&{+Qh(KHSUXGNqFV3U>HTatbRs5Lno<pP5nm%YDtKVANh;$Q;EX#_DBE8UkDE8T?|c#s
LjsdpRZpD}T)7=*zF64dLR_+B8EY22+f@V<4_IQ@)b5bovOoS-r<9r6`5`v1w@aQa4c=G!Z+afIMehUZruB#Z;J>`xpme+h4}^T9qFz|xG<DX5?^`eMSpEBQ
x6=(Lg<>oH&fEbxr$M)FxgKJzmLnY>=wO3WoQ*<F1KWH=@&gV2pqh%x^j!6V>0Fqty@CS_YEDO{f0n`X!Qb+a3j+|i8oQnrYXf5|lEF203miDuXTyKV2*3Vu
E}uSp86SMysYy-r@xpiM827Ox=sqwWY9$Us72^a!aZ(6Yx0A0NGcyM9cN4*Kx-#%f=J5TzX8|I;8T)980Q};7_@hRM7p0%CUeKj9#KlR8`KDYcNDi~tDNQ)Q
DVlwqQpXLCKeyT6G|GaQ{I4u~UIFNYej#urw#A>sdXGpV3j_`H=sc3J$2DH<c#CjxOkN6}xj5hilj`i;tqBvTw@gbMkdTK3!LsDfdLMWnznfJl8vs+D^D6($
lTp8&DLBv`0o%K8-&dK1AmDB~#rkj<p7Hg?unuKFE?4d2MpqmfgVNKT_j<iVp1<>TRvFat&>ssT0(Ny@+Mo1y1M5g&#dKgai2lh?TZQh;pNP@-1XX<q1&`f)
bSxSZ1V!en#9pXodGh?OYy@Nm%LkgI957DtZqn*}0wiwE6mt^CLNH1uoalNq1T_d>y)jZi-ix=lv|XJb@WZZ=<BcW!3Zh$X8#uyy=Csz$N*9dzm(z%MxFMzc
`L(665HKmtb+pIY!8w(qE_EjoF4x>oU;UO1lgg-YwIp-2lAJgx)nEl~UKb^{W;J-3JlNcru|P`R2SvL3PN3NS({ksn1=Kb|4*xXN!83*)ha~tTRBhufOOY-F
bt{?gYMLSJBgy2JhB*WRz9wH6OM_Zgw8g?p6BI>F>XoDlp+tS$yV=tSwXzvTrMIG?Zr|$LeIx@~?}dr@$CB}+&n}dp@h0A#y)Iy{k_Jc1KQwC$2Kah^%Xn?w
92T6B-SkPG|K*f*s?PRmx*7OihC9ix+Jiz$pD7~A4nYS$kaYX`AfnIl{?1VYI991zWD&ZcB#Qaorx^)+RMr2Gmurn5(mcB>pY;$H?-1uVoQFlXHEJ&=eN?|?
n)5EU!^eX?Qo}#wpm3tYSx?IYI;{st$jO4>>gf9DmVz<D*8i4Ydufk|7q<Hyj+w%gXvFW)3kjSpxbebZ#T`}umQ#Q8--5Nb)9Jg8S$IJb!eNy729fdf;ku$8
;GLwMeeCau_T<)-&jq^B^P(A5VoFAsl>TcjlN_YF)gJAyEWlmG82U#(Y2Yn-zj9E&0H%g*o9w6WLfk29`s@!kG}p3RCKUEZ&C@@b52Q*#ycjY+px_HZm#1sK
p-IsHyJ*z>BnU30<`Zugj9~NPNd3^KP{;-4>{EN@31RicrvA5an7P^fOf2OFIM48V<;G+o-N&=wWPk^_)tb9jbxJX`!&}ENTZ)I$edLR6-nf>M$r?#oiq`q3
hH6tD2+$u{Oy;gYS4WKGtG#=4dbwZC8N7*oR<9;RiY<{*J4X`}m5q$W$ydHfMR;l#KSO#U6D<6d%PUFc$Uc&KQM<|()KvQ;_G_g<!M6YE%hqH#E&4wle=38O
ovm=Tz+iZWPF#}f4u!81QI^Y#3|z8N+vxfi2Fr^N*xIf|fq}i{S9q8=;(k9&B#3mzBjxKKl0RC)?uikxD_aOCnVa0toi4=>`LjMddwp)9L;0OpXfb42gEXks
Qo(fiN^SR)HU^<6Jye{ArS-ebk?}=1LX$Fenx+8V)q=J~!{NBV!&7;3AOgy#2HCYNE0IEDYiuX@7*EF{m(<MdVcf5HDk$a_G*VfPq-FcVQCda8AkH0`a_I+&
gOYJ>TiP+rBM+2zQ|_xj(vWEJ(M`lO97AEJuRrO!g=*<2p1=dj=%ly1xh!mpa_-gB?*6H0<#bx<G7kr-U0w&xm^_qw85g@8vheK7IX7?1RJ1GA=8ybyhEK-N
x_ODa&=C3e`XHkZ-t<eUypi%knTWf@0hR<Ljo%1R{gDZ+wO>!8Lw(?LF7gBYp*YMlzWZT*!VE9+2{oL9varfa)#<WW2QAM&*3;36AUihuA*7@bJd!t<gQycw
++MWxRN5E`6}fpiJ+YYhn9<TykcM#qgZl;l%t4~ragR!=4*7X4-#uiL(dTftrQxG7Rt(l04YXXbm>1-xN~ntWC`w<gG{ISt^L=01jgaFrxhIN|xR(+8n#bA>
{v5~pcvsX=djIVjflD;lJS-lsJaGZF&2+y<bUt3YhHMMk#iDlJj+~Y>7QUq2`lM&lkl84Ek;C#GCNhn*si)e|eY*X<sE!FN%}Y!I$BhxMxJw`*pMs;Oew-mK
)`VmBv-1Yt@%WfuQFS!N2b%583h5sd;3OF3m8hHs>Sq@IbbsT)V*Bp%8+8%<xUF!!Nz@Z=^TG`8`&1z++1NZMWQDNXcMHSGz3}&Y$HYMn6FB+?5L?qr!>rx)
bME^aI1+Z3XrUnkHDs0XHer?^{=#wHJ~aX#-}SS88Usv}sFN6@?eLPF-%#mU8J2CHTHEJw;@?CsefE_*SZnVwI7!unChv?VE!tKPuIk;G(7%h~_fg*?f7(D#
-SyGc)j;H(KYi$@atsa};iKVG427qhKUvdDUA&$7T$C^90foxpp*R6^+@3#DV(S!vUB2Ly*mI5;(b^Hq9CJcd-^a%z!jg#LJp1AMQ*XHKth@VB-a$6ip_eD}
{V^Py;Jnq~0aB80-KUO@q48EV^A9ntJ^wo&OUoV$O|p*nLjoZnCUrg#@lfhNesgHl?6meoKp@o5D<>chHTv9D?EbY7S2{>S)v>o1h`)?S3y0#s(=D%Ny>vK4
mAyEUa~G|zU#Z%-xnj!j@B{Ocu8=TG`_SvgjL7Sq<v-~i@q+8MXEnPY-mGq2m!tH=wpVk*E!HGli`p03@X!_3{wl&o?;P-S`b5g*26z0-zxMchTL1*lkX*^1
@xi;Q82+)dQSdtQ(Rz_06x5{cKgEWuade?sNXfw!RlKK8O0yfn<7K%p2dfYC`Hprs@kZbZW6tsEJ^$h~wX=A4RRfJH17h`ev{5~r@qnPi1ojVT3_SI`P^$c#
U2Vz-R%OcnnEsYRNmb__i+m)O!{5Y2FD61PC~jo^bTSkuPOPquI^)vtr>^7$UkFLd)pz}nhHW^(TN;i%PQKI%@N9Jh8^v<Pz4OJ;TAFTszHSeK@5LK|8R95P
ICJDdeGJYFKX*2tDM947!lzd><56(W;p&A)SqRo9vRS4HNAmnZKlxNEWGLk??Ywlw+{T2NVyPEexc?n{Y@7_P1MwH68!qFBxb2^NnT|+M(voOeHiYRVffttN
H87DVOW-ibgQ;`XZY0s(Xyw*59qb53Ob#6lUy9M5Cj{*hw#38M$-Yui+YzsyhvZyxjKC4Wpr@7m0T6oLG;_VW2r>)yhHdV9952bVZuk<2-D$t&KmYsxt9v%G
(|^72wJtdQg_Sop;y<1_HK>IA?-#BLZtnHliu})S&K6*UpgmRipDC{W>a=>8Yz2=^Lr;t6spvB3G;V$vjrXsnYiui{5FNAHxcf>1f31jXY8UcQm958cKBo{O
+q4|1oO^z$^!9N_rxLDeW!aw0cf)7S&O!BTbNu!v5jRrw#Mt%;Emj$4{H>)HA2{ZXx!mm+1kYTM6TtJUTR9wI3%hL9d!C!9LJk|YWbmf_JETn&hI8dV1gxrp
K=`}nVZC!W_|jj!-XRM`hEaA6QE?*FXAcuL3k2Zj`KaGS)oGBWdwlN0=O-X_n@n-mu?OK9OJ!Yp8&EqAgf%fZLR5BMIy^N5+Lz3|>Yf-t(DdK!bPjh2yr|sn
Qwu=eqZNqv#382i;RBs_H{cS!c<@AA2rhFs8)z!}LM#4&^5wgcP=9x9A8ES>f^OvIO^t*@mH5vq8SUi%a(eX8`z9l5A3TgG6TCc|joAu%PEGYz<P+pSC|?Ug
FJDu(qJk-?BdK0~IUfa$eaY=tx||Tj<@)8BV*(nL(%&<R_~6QITNc%>B-n`jDLyP8f~JF7C%VRrV8RhOKOLe6eW4zyDFtU(tTMHV-rBo=LwtvRM+ao@{(QL>
5(Ew@mQ=<CPc(jOT9p1FfYhQ4&6gSWcstwohk`Q=XGNy2W^ZOf^^(D<s6;Pt|JMHyv|s_zOE)=nQcRGv@phR~-VBl(E5=&kAxNC<qNX+vL)Jh%?cdTk@XOC%
IAGw5vx}jxx*EJOuV#0h|6v?jygBG<sSUADZH#P)+Xa=z*}uXvETC{cO+?($94|9W9aecmQA2Oz?$%?7DH+;chFW_JuD@g}GD(HzNXrV_xI03dw2M>J6L79T
%geLe72YP6l^F$Ya1qpa|0p#ay;MI<4a#lt^zY2`9=|lWxep#}d}D)W=Y%$o9nZyKB2xPog;roJ(0#u@_%@hylp59yLt!zA^i|6nxW?EP@aVM-D($bU+}Vu=
ferchzs^ypH94LZWEYK&IKoKYtT3z=dgb?OU%?vHXnjzS3%bIF|9zZtM>~Cocf0;2w0m^Clr#>2O;Z0Ep>+#*xQ!fAI24b@X{(r4KX)|RDJ^?{wMV~V7lFxF
8{DR<Q2Ux3h{2$qqQW$Nl>ZrbFW$Q^**C{OE=67l;9<4Pr4K`WE?-@(T?9y$*<yWqEkVw<+BRz}iy@`ivM+w25M*ubm`INR;cLDC_q|>qB`q%(yX**$_@H=h
5obvAS<EfbWaBV*#r{WK<uGTYG1st(N9Xq^2`;Y#P(8{+_;5TP3|<_qS-<pg@v!2l5CU&FHSoS?G>pXGqi*Nu{&<0%gY<A$nKvX^!kdb<GBNkDv`{Ti7ZEi0
GMr?A@};G|7dd-;=KGa-=uawq`7beuQOJYCpT^jwEfE4bJ&H2+wjc_d3=`ek`yPwm`YV5C!Q#r#oLAS(aegd+|JNE@R414X<r@28-!Bu2i}k@Ewv)}5{AvwW
BkRuOcY%0LdxlUU#RfWJqvK)?ju7IO5ubC6M=1OKhp#W2;5yG>dW>lR%16YT5A1Q2z$d0ZzDNOOWg{6sJnV5$EOFe7Edlw=$wsu-91*A{b~)EI3OA$0uU?$<
1Cz4CjeKWclwa*9V~j8b@6dXL^NS#iA62E3S+fQ&|8(vHV=3G{NB*|0)D3=YPv#TSG*G@~7jHAyfPtDTgN)@#AYS@>>h#$()Kt%HeI<;+Yy<5;WRW>u>ybok
7AeA9f`r%HGZtgjK@TE~U68uGg$qP^Xy$6Lc#&;@r4y`|J{<4?!A{0`<!2Fa@KDJ(WEuwHHnH7+DL2qfgig#}@q$5_YdSZL0Jfy}Y3NV+<KCuG+|;lu5<RxX
<BUW>N|W(8uG9&eOCpJfY5Q>IVKmXxKXy>Oq$60N7>#TNO4*v)>mVDUP1~x~g|>9jJ*D9YP@a2Da$h?d;t8Tce`cf5e9-^5{+tQk?AoU;(_7$MYv6CQZ9`l-
JXbBn6@`XZ4NT0VLNJlP@JC4G28i6U9yW_-fJ9x+!8JP)nj8MVO+~}uR}$Lxhgug4wFSi?4;>-H$=|tZYljO453e#S*ny#S;FRT}F{*Tjf~Ovt;W}}qmEfo;
#4Z$)s?nxmZQ-K8qs~y=pv$4B<qSf0QU&+der=e=h~^)O2*Ld$7rX!3c|s>bk>qf)8*1fk%n64gu=qhpaMsom4~<;NLT@<3m084A)x;JoFQ$6l9KVaP&k7_g
C)obuHz}2JNG)ds=C4*gcyitq>F*oY#`Gc(?=j}$vlNX##ob<kAM7Dca&cz)RU8I~e=zD_b%2n);oV=MnwV82USLwS!?g5r-ARieY?(ifF>+7B%eNLylHGg#
^ub`Rjj}I3WamjX=2wG0nO$QlCK1eoJ29s_6LDK;qvcsoC?**W#z|iAh4tXY<Mfr0`21egGPf-r*;O;g)sAQ2FJr60>wB)aqw7q4jlu?vk>pc+v({)ZvaVYb
%7kN*%a2gfbjVdR3GO_-jqfa4L`O)Iq0BTm#J|^fJ{rv45Zuzn1@G^j4cVS>F4>yTNb`h*#`Nn8nQb5w3dmCRwgHiSm7k1CF?3fm^a(1W5%a8Z!kQxmJH!2X
0=XtoNen)ImBt<cTpZ%pxzo|or@#14rw~Uf*Lvsv1;d(QGWKSJ15&?l>k!|T#b;`T<m7iH5K(>bp_BXt7FV@`ZrmNg^w^O!r*B3$e?jl)!&7nS?&Yq#_2v$K
muKy(G0j1{_wTi}<vj3H*xR=I<)Gp7$uN&34aC{@G&@n%-~<<EarLKYkmo-2)w>o5%k%5p&jxJa%+T4Nek~WHN~u?}|AnBNo?7DOes3gSkWeR2^MP9ofqtxY
INlgkvW7clK&juLvTDzRs2_??#*)QDNGh;GVks9P5fNs_Zn>y+W7B_<9FF@1%$;`<vhkW;scT<CFeo$*80r{m!?|PU)69`Pa2~k4oOWby-w-<p4exhF;QKY+
o{mJw^$>qK+Uo_!chfPt+IPVFVx%r?<~BA+6SKKb@Ilx(h3jg07)tipdAWv#!_unh@$zymnySvHef_Bi6_+PvyVreTek^lJdRG^UlBakWFD772c$TtAz!E<#
X-={cdg33c>q6M+1iY19JTBl_45QcQMBfMoLHb3n{I8x{I4fgaI3jBXg=x;*-|}@B+c`mTEzTZyZ>tHM-#?5u+A|YUAA<4k>U70wa4O10KYVZE&cy!na^HIT
ePP=i&PkFVhO6W$9#(<DXc{?G^d!gxrX!<_H}`r4yX0>s#wQVoo*%OR8eoZmcQ?(~0=(h9uS)Z1`VG*W-7V(W%E6zoj2p|^*<iL->pG|q1BJp#V#m@fyuS4}
*Z8G32z6914sR!bUtpZ|7gO;6f7gMF;9y>y2R4@t#;3G`(L;1*q9;BX=eXMQpU(#2j4o;8+}`s|`>$C{o2TN*;5qBhy0)mfC*GcQ%@-x?8UtI|UWnRmYRikr
K!xl5r22X*WLAX@)t+#~(<A;mU!&aNE6#jUAus`PRflCn$nHU6Q<Xp5!5!}J8MKb=c}W&S#XU7?SMV*<4tPu%<M#P43xdYM_{;5Gsi5VHW2xsan{B$oWa5@o
Es-lCx69i_tt`Q=)n{oJ<OpV`7|JPS2XqAM-=(T>K%>aX%j_o3;1$<=om1(7Dhi>veLrnsA3mzh&u##V1tK!xst~Lef2qIt!2@EdOs0}M*7)XS#2iy)k9bB)
u~%Xis5b8lUzD=P@=4{qcwaSmGiI$ssY)Pc>CO6?`(`*tmiyW;DIA<b#?C384j8Y0!P47j4%fx?Ben}}pdgd&iV=y#<G$c=?*Mar{Xy$Ib0ZO^qkq_?(h@;D
Ri3GJB^(w9;-^d6d=Om7`&2n23~|EK4NqT(Vba-G*m2VnDwAlv`!5dE7LC5F>|q%G%{TTwAQ}{uw8FWZ7KpB^JbAd#46|(Amb2#skRqix`<){kHu{5u!4;8v
=Un;B*>V}{!wen2zS%&^w8Z3Ej65v*R-nI`j)>W|NBNZQc(9hDZ>?vFUkA44zOq}v%vN8=fzScDROf1o>!hJb9hah_asv-NN!{94{1HUsXAu=-0kby?5&DVt
AgwD{tXucNBRX~U1X*zq#)(uvSN6d<=c9yY9t2>Jb$;;^sTGz+9$dM!=h1(zlsOrAyCXs_SYb`y8N~k@2k3mwpyVY`A#AUT`khyA$t&#8YWr0G+lC{!njSPX
S;gatb@-)<Eidd>35Y-b*bogz8Jfzp{lH@TbtjS85tT(xq&Hsq!l_g<WQ|w=I(z!Wf6yBee}wAeQ}*WBo2{ju(G!XSM{83|`S4Km1=VNCAbh!djFq|C4S!uU
wd`~pz|XV&$x9^`40;R8|01Liwk;bXYhn*;QkKhgk#X3Jh%2azwMF~JPz{-)2RIj8<2qs;aM3;b9Bq>(G^S3?sV<n|xpCJ5os=V-{kxVw)JR~{MBv5Ang|Hq
V(L=AtcnLLRF)-Pk|>2cuQ`_+5*f~1OtFuGn9m)TDjs#D+cZ(!zvl>C`M~!1KoXj4YR~=V^T+;QVuiPhEnr%l<LO6Wf&yEcUooSpP;;?+>|3UTD~E{awtB;$
u+)Ds`k+5tV%&Iw*Y@^HC#h~>XDLpIym;7h)fmB^1SXX0uJGjO9d^`mKuu)h+FHx?y`IQWXe8l^0<OQ-Qu}mpi24;d$1NumXXG5||0s$&YI*IyLcX{zvr&J(
$Opt1n51vE?(v3jaLD$KC6?4~RZ&l7Ay4xkp}GeTM6&nE-(s+W%ipul%fx(e+~Da&-uX)?RdhP16%v7ii)<Q7g70xc&FECvR|~}Iof<5edWNK6>_4s(2N}s0
qoGI_*lQ}(o<Hx4uiS?TK1`>exk)&}dT_4~s^$KCaxey$9x=!SYCFTzWMDb%x;i4|S;^=4JW#X{|6T0UMO2CMyY12iz~zBrDJ!ub?yQAHj%fPBu6xYtLWe8n
KmO)Zyf_Pai`%}uEm?5g)@KsUjKYJ3iC1?l)ZkDYt90C60rlQzjJo30VWh1$cY3ei5M5%5PQMq2)7*EqVk&g;++UohATb5K+pEdmKH(U4^|)<$L>+OKdU?VH
o*1EebNZiSIHG*s>0GLGg2|bX7XQ6o<1lj9_bFEielm<jg+2?!;oFHdy4w+`*E`hvCXMGmeme%m&ad7N0@cjL?{zDCc~__waUZe5M55BmXS{~!;WyK{WgZBd
*wb3N6oJ_N`l+D8$_Z-a%t33yZuqsOAWRkzfc7Wq7p`n)K-6F1sKBZg>NCkd`&QY3DxqmEjl~S}=jlJKN-N-vKJAiIjtTAs=(zsK4~JF8Zk^w(E7C%};zpS)
fHTGs94EB#SL*R@YQG2W3k?}|ajD}e%>!9ARc9EJ-1#0UWRIGMYiz+Hb|_t#mAUl98p*t+4>{}1koI`(mkz%r9D4=bUQzGm;iA%YW^++|rKsO;!@`dQ*3LwV
E<OBser4GFx(VhE8cN*})WqA=<)!5v9*7FkDwP<z!A_Z*DV%%H(?5^5Bn--c$#A8tO9}937e47+b`iPqf7&!y^f35{z}z;{5fxF7R&%ZP=En7!ol=25$S&DD
5-Sx$%%Rmcv?n9*{i8sz+)p#)51$`55VgbPU~@0Y2pi@I4_}#mBn73T+Ps8(GH_P%oUvIkLQb$POH-R3hU<U)Y`<&=>J9#*Rz@5+&ZJG8@<$%UYq_iz?{&by
_vxASA7i*)Dfj=7A%Qp5haGaF;~^jyT`IB1g{AYghjXRnaKirfc_mqGh}R`D&~0nub^7B$f|Nj99Til(>}Lya8yQinbsda-8uu}rx5w&6s+^p*C5B#o7pnSa
1!Fss?-Ufy$bS2=>SUWd1gM{+8@*u$N#A8^u{CofSZZ6(P?|t2P3J^_yc#%dr}gNlq!AHGMrHHY1#d|hX(AoW(R<@)06DiF;!pph&ySSA$pM0UFN_?qQE2#1
THg=cKkWvr=dBU=dtb7Mf*4jjir!9c%D`)fxAv)`7C4SCcUFk2Vw_+j&dFLCK5j4TRkBs^zW0>(#5rA5(^Ov;j8ex?1I1fUUp>UUT(~lq=n4+nUH_`HYA_VE
iD^r<g}G6nf@;Av@Tstw39KoAj_?!7u96L|b2u`bn%4xCGm(HArxIvNE_9XHtAN0-ZY|@yCRo4Pdonv2Lj3F1FGV(c+`Y9zmF{DPH<_nS_-pGz?Y715OR7BZ
KDWL*5TgLSie0KXqrG##Ix4PJTcNT3(`PSZ1Kg^OW~osQgNs~d9`B$dp0GCfFW=i6uzIqABz;qO9;#YZo>u=ar*)Bw^=z5$kiO^Q9Qi8%skvi6pId2RKf7Fp
Nv1iT4uARPU8aDwV~QbXBJ}ZfSv`u!F%%h0m8D;_lu$bq)Zj9q3Kmp^-7ogUO__Zr+udeJRNzA6kRbFayljpUTt?vaYO+={XB=Tv>~<rz!k=1Ve*0c&*sT)y
l~wDZd6Ha%P(TWOQ>q$_H-)gfbF;LgFd3Q-2`i>X9-!RF?wh9f!0)t$f`)om{FDgn`BQ9#kCD?}870c#(f{>$HrWESvS%a7+f=~yL-4H0r$Ed<J@uK0(Fu)~
+b21=?7&SHZa*dF1}9JdF}heYYz>DJ-4b?ztWU+N@VGTPxM)w!5;&ndFI6Zs-35fcKa5NFxWh3|eU|sN0>~r3x<qkZL#RZ*P}1WNL|1GNc222dj{m`;Q@1C$
n7*3?uSZ}=QOzx6!w6hc648OL^TF!D)2f$i3@VMbo5^Qvkt6$9$Zek{ioF<(&&pe&lvg;-xW@&YPtSF;$hxD_BYf^Cw+o26Mb0@#2BD+3yvR>n0#)-O!3p=g
k$1=R-NWm9{dQ_-n4!oN9~haNhu>J?)9sCj-|`a3B_a<WOf|*;Q`=|aEeFW-IbXgmBMigJQ~o9|{gF0xSV~pI6-g!4nxz{C0D1O_k}rzz6u)~yzDfgr6W!7F
hs1G`vc}=Km?O?Oyf6_|l)yyOoTH?;Cn!3P3LUU_#oA~r-RDcTxX@{SRng5Ahf13M(D8d?E~0w(l)D(>ikPKkEgUc{w>F*N<B0@^j4x^<&ZuKy4zKsrhnVKq
ua~mCK)L-gA|=Nc0c9h3XK2mQ^6c72g8@f4aap<0E<MJv%}sw<MrAx)@FK}<zXeu3;Uh*t&akn2@U!=rKGJ@Ag<08{!N>gx_t<_nTpakUMK^dG4cq_9c&DWh
?D5Hy*;5{7_nLk01i3@eyfP$C-3BZA`6E_=>_`ifh{-y>=Uqq0^rnklF-785Zdr33Y+)MRVQ1|T$YsZ>qAr1+E`o2ulyVrY4$TvIVgL<?QQganUNC?4_*gYh
6jE%4r!UdyLGB(u-IZ`b>=WR8eWSby>jqz2xo>#jo!OtisqZ4eruvRB;H5jtUT+r=|29LI9e26DqYQ2x<W)IUs0~e>s$+dO_j1XU#_*v`1h0ONDij=X#Jh#%
;kaKGXe^Z)A`6$nzDx#s3pq1**y~SAvvJ|=&3#+{d^JG$ZuG#;v@}*KH!pejXdsG<$ecFX6p{CnLh1Hjg7kw%C;fPR<ox*^B~X7ICQ@boE1ZtFNi*!=bv6u*
cW#$hamc`zw;+y>-xa4G=QkVaSRgVzIOMUcHQqU7Gn(HJ#<4O{_MlWtgi_b6k5ldC|9lBU--s}@e?IgQTQG!5`dAP7qz1B&$`Zc0FNrgUU+*`WEXJ2`j<qm1
Zz!JdbD?D9g|fxt2db=oIL%7l9d5;mh_h3w1(veVa|z3&h9u_xG6tL33ZW(V!pp9^{9sIY7#U>jjRL<b8^)5hpmh8-d2vM^wIS8Nl-T!tW1ifTGl30s?(D3=
ALQ}Qk;$T|J`L0EcJ$;Q=@4!-8uvib3nnyW+GLaVs8hL~c+Nu-dei4=lIc7lV^nLaRtO|K9uZm*u*Uq@M)-8ICY~9#ls)kDLGI!wo`EHC92cJru6>}1cS`!B
lUpj#lIV!~uoHs^n-pBa&1yLKHAVM!r2}Mb4{P)1O5)kKxso46R-oIRh-s{fz-oi@mT$Edvf8PdcF4^zMaDZVyT_BNV+D!tUulBx$D(5|dl4>82F_kJaD~E{
HFI8t*nj-~mv~V9$s=b7x_o|<B~}12p`(gX-4?KsyDGj_vzOzSOY^Q<x}fhmBUT_7g%h-=3d5{i5z;}V*t6)2p?y<7dfz63%hVuQsE8Fd=T2>25Q@jz^HOK7
AEB6&y-NH+JPh61HR2x{!tk%>OMabWGQ?%I4{`G(f%q$LGu7`f)K^hSIvsTcPpiY}a~8JvT*PHv?&g9Bd)fn>&xXNZ8-6Zoujf*KlRiW9(Evs6xsPdVZ4t*m
^{L%t5H)_9UiaIhAY4v;jNCK|)N6VQGUnk(7qnhkdLs^lsCwP#y}t2Dw~c^uI}$AJmCI3)5m2t%ab|XpMQi({)P={kFiClZVXm79*m|k?%RC(v?5|&6Y>I=2
i153uEH?-jb7_+48{yM|$~!_b-tZ4nQ=vJNf|unJKY4~-VJCc7TASMjM{Qf&HASzZm9#zf&z{fFkKSBAkQ|DLMz$|)!NH&_526#OGR31$;&Q1!ec-1#Sdz!@
hS@E8zW&i@JWU8JHn^OL&x_=a1I^*se?n8ubXyiRa!T_S9VNIH?qw=d<c+wJ!@Ig^*4VtfH1=_y76uK@%_-NXV(U?#^vC5GY|8QJXNp?E`Nv%%f?sy9`V>ml
sqKn^e(ApxiMK&}YmCbDu06iRNzy0P-#{Y4d+qY5MBLoJpEHF&4Q-R0+LVX0F#7mpS;<;3Y79lyi+}F*0&16KLMA=LI-6IBk^6(X^WIh012LdX$Pyy{Yk}bD
<Ma)3MwnT!OG@4Gg#J>8+kwGEjP`Q0zCM(SvaQGVosL)WkCV)eusjC4*@AIBX9IAP^KI$&wM0}kr`k6}$l%FAn~zrMF>uLs;rN!L2CpdtzQ40!7#%khOnYsN
nPJwT*VPjc_PzH0PpJ=@NPJx+8j3Oa(W2a%@&;_9Nd`W=jY0mp-W{qIUo3fwGhMMR0;l>9!q$Cmcx<%!DKXju)0Jy=T=Slo32*xmeefm}5--**lf<J*fnS~{
+6K21TqJe~z2SN0FH#Lnpy;}ENac_@m}_TkXysB6PU<;+XxIgHj+K}1)9c_?qOXNRbtY(SBSHhu`9SaRhl!?~9Aq3Bul*@_0~_ZA&+7-f!CR&6GQDpA3Ou>&
bQV02xX?Sj(p?C$3JdEeY?1%vw7r-?%hl2yhK&b09gc)S!0PRYZDR_CJ1q9C)Fr}~>&uvzvkl^zqZw9i#KYy<4@^6n!qDXYyJ>wlI5ueBXl9ATQYleLz*HJe
UOFPwmSuyFYx3b`$39`=cuc#Ik_&{@737}|rlDzRw1+aF9^I^8W5^?Hz(_78*fSUoSr>&DDtDal%04sh=WktPM9hW1>J7nDfBE(qQ)jfLF{CGT*&)d-@K^5#
58Qgj*7D|!0Z5aB<S#%2>m%%g>)aaHj^da5s&0gP>ro0KcVbYRz~8=E<b=-dwC903d;8$8>+2KWRPn~s`NC^Ydl<W^ob+$l^9|-1*5O)x@Q&nNXeHFbSc#t=
Tbl;T6e!<)yQ~BE>$h+6*@`2%->5wGr8Vvcmmg-_%k@5H%U;<|djxYzwJ7pg!ph9qX~@kD{*TpPv~z1=s+Ku7_>T@A6CUTd)2#>A_qP5sy`F%Q&m<Fp19B~W
`fO%R5$Yq?+q9SerrY%ztT_faC`i=ztH=&pibgfdavpfvhNbCYTgcs5Vj;g{hsu6NrShNVNV4c67}@WHpmC{{;sb`D4jfQ<M$d=dZk<C;Ja*`kPN2J^XOA8x
k826^JZRDkXzPBjj(pGKk47%3LF`_q#R^q2(kLlbqbs!`=}B;KrbHWj7Z{fh6{%t1HbMQl8?NY1tE1X6lR%TLTG@dyKhTf8@-6Rih0%*0`a<bI+<rp&>7}>@
2s*kBDjE9Y^Q#4p$Is+2$(Q?2rQHN?4hAp9PDH}5sXm{2RU1-1f93>^=^^l)q}PWAcbq)on|#~D60014Uv#(VVl!Hv$o`EA#$CN%W*#xc^ogZS=XE2HJL|JO
x$lRefzv1Mu$h2lsg&`ggEfddp4bL-Ss~>e*Eh0L`lzG4X6tv(2KSqXDX-lSf_ZGVvrW1#+*`D_yMpY%)S)w4u;Ge7ha0wO3x#mKDQNV@-rkvASbV-;!VM9&
b9)5bJAb;L!Q9Opg~zycpDfA2=1R3wUy&!e)~6-fcpbo-!bE!WgE=NdHpyxF#b91sMzZ8&gf~S7q6a1vF`DY_ZA)wpp4S?`-^2%@;yvYMmzJv#h#X9dD#}KX
;_86Bs04mgBzpS362)pg9mx)-9kj+L{bP6CP^8|**ZM>jfnN&pv-Pau#2P`u=<k6*l9VmZX8~wQ{2L-hXo#_bxyNTWOkkD8*YG-K46l<72G}Mv&^fno^_BAw
a4q{l=Q&TXF9$Hk&?;kXo28OoGYz$4Rm6RVw6IOqZvMy84=ac3*q$%iK%CAkzB^qT*J3+1+V)w3y+r2(DW4`3f(90^u9)FqLyCuFk_gt>zFemqPsNvh58a$g
W}x%p-;Lhgd%l_Z`12i_2zLpmsp$8{Qu$GGvcgL+pzO>njrM~tquld*Hnu1i%aEKJ5XM~}N?QH)=a5bde(LZ~5$Y|8Hq-1z@Sgc9^vx&=4WA9~xH!6?#d3V6
dPo<mS04o)VY7tF-Dh>1$30O!=-rlxD5%V<s~l6Z#ko4>skqEQbU0<l<~7KokND&r2lh-bzSzeYL46MIxEfUc*;(UG)^C%nJEo{fXO2*rb4IcikAhCR9d6qy
q#ofg0onbk?JhoFB$;hwMZIu<cMW$;>Td~*v7|S%y%j}SSdQrP5ha`miix~$mI4R)YwO=W$KZnHFK*?p@*wUy_%Y5b1Scwe?)?k%fwx9rz)-6edNsOr1$}LB
Ahh(F!l?vsGyYoL9Zv+&PFLsZo?j;#g-pL(i^SQDkk}&+BH<`)Dt!Ej2Nw3L5X$P%|HtnE_t>{7Nx8V<dU!Xc*A<Z_RneD8G@!V7@5lPG7EbN^%l+ex8Q82C
SDreX!(NL0bh@`QZrI&@QjzI`Vcyu0h1N)f#q#=_CECKBo@u8s%O3I`78j2uc))XdxUT$E1TMCV(jBOXfNj=+_5g)Qd?}Lm<z;ik*vV<52}TcaN!EtV^F`pQ
ZLnfalO1Y}A9-EN3dQt3(XcLWCx}c-cQ}VyBHLz=Vx1uvnb)eN4p8c&`QTB_OrdBbzJGY~XjmjnIee)1y>ddIP{?L$vkJ=D;>{U}lwo;wQJ&(YCe)9YknW3$
#=E1k;S?GE_$ShHcit^=uP0xAE4FC|z22EqyFNA$<(GXG)94D#KVaR`cR^JDj4$CqH`qDVL<=$R`NM2%<f&t6P;K4qb$ITK^g(($b1i?cDV|ScDzJp^z{Y|~
?w)6^nD%`aO2y2p$Fh0d;fVLsA@%aj#qfTP3IPvwoWJnu$8%M01e$tG+U6VKD#P>J5~N;;4qJKJe>4^E?aOv#(^K(`<NEvFd_J6&q+!zuu|jq~SxA1j5e5o5
<u5+f#&g$Z-J6xXAa9z|Fd?(SPEdg8SJ_a!6;XS&QmlgEe!d=B84vKhKe+#EkS(kx1CH#^r~%FAl}m@>5^(Lrjwp?N2&NMF74v8FQKd8fUfMDpeLc}EyA<Y_
d|`O$v~DcePn$kb)(b$hVR8RXxB}+)K;xa^2xm5~m!i6%kUmX@A{{-PRj$l_5}<>Ur2V>W&2jj|I&-9UGZdd2CL)c)(h)TqIR8LE0BP%$KWvAR;W-zha`#3W
^3AmhRT}f~NGi@_^OiI&sCj=nZQz8~=hPy;7gC|O|8^*4mjgcZgkR#w@I{|rkv*9}24pk7FUqz1A@D{_+J$XTXg?}+OYCrhj@|GO{gp%*X+)^}+VVtnaCH&e
6-$iBBz+)b4FsX|b@ni3GgK%A6yNI%MV{H#(Z2!iD7CA0uPTWH?H8v=N;@O4J>kwfpJffX_aA&E<&-dy-`i~SSQ9d>gZaXHo}fnD@}`AF0mhTDlq0p)s9D$X
(^v_Dz)YHXV{{<4hZtCj8pAPNf76@$cPPFQ9{ytTHwvOLFT(B9y+O$&%_6kojQ*K<F_!atg66^c-JZ(vznt=MSBUf}1fz12n!f(ZBYbJ_K0g215pwke(}HYX
5Hj=}vEi~sO?V>p^WGSkf1u)7nlXV5iDLD}vO8YrjNfn=^~5$^&((0N7|<79E&ipJ2l}IRMT|o_xP05E@Zr2JsPa5mx%ymjCiFM~5l0o&B%)Ti_x8m>e%Ekw
?{MfJNfRkk3rE?_+$(1BSMl8?Rmft%4+`SLYq>!WprbXC!oLv-i;m7|YCbpoIZZ(vMr?_(p$v|b&&)9V$p6O2`?@$;C-dvics!Ih<>$O=+(6dSGkbD06iFXD
1`5~1@%xx5<24I+=xqkvElbLQxw<anHGfmE1~e9!w?;v;qvKt>O$YAliN!ti_J$?rDwSirKk|&-{t2E?K*H<$eI*_`I5e#LQ~ayr9=D52=4niUn=NN~7^Bd-
apQ(~j~h0uRNFlFo1mM|zi%+v9)Ej<r8(%#k<DRn`9MerMslaG`I4EzL`{~F@naPD8}cl}c6<>oM%Usg5sv!t#N1k?NX%^VZ6!}7Lac57=g2iPwAPNwuRV*v
%ffKu!?Zrww_kgpBzkWTN(dHGzx4)Vw)I6%&o?;M?{X)g(jH45Uv;?3QXq3&kCx&UFZ?_8bYcpv(C~Y?r2h6ZP_9-du1ws-_$hJ@OHFI+6R~O>WV3?!KZB{l
IZar-7@9a6VgMR3XT|qPy5N<ixgFKuf%2Vj<NXf)=s%uCR2XZ3ihs2XZNDQ>92HP>X4M3fF^AiVCnR8=F<$E7XbXXF?~X^@iNj=`q`^dqEuvig-KHOUp<N`>
UhBIXE_+m-*6ELc-TJ!!fu1M$6!;->H^2lfqCX05KJ|cZ;KfNgK`XTU5%fJws|&BJ5#71x;ZXL<wh_1b-*-7C4P}Miq4tfZ_S<qhgz|IcyUS&xDu<Zms)#PQ
e(bPWYMY~$x~=ZKq7&W@&se9m1S5N0mhOIzI*RUodt@3Fht=p03H*W{$fovWGy7o)37>(Z4FtKMC}eA#{ilcMG5_URD{ai+;k=d5o^P+*8IqgyhoayyQc|^0
9Ly+PzrXH^QxiodS-ZNR@V+PzZ0v;7=X+1}`^SUtJHZ1xMhjdLG#?f2hydNj@-T6S2i||L{&;{!63cQ|rZgtAp!2DTDNZg9GXxjqR{!e2rGi!d&`DpU?0R=R
w~By!&4q=NdtBV`O=pyLFvXEu1%yS77AQ)(Ew-d#kGeyox9ivB@X+n++u1`7$am>rzu9Sq4e6WB74zyiGq}DPYUcx;xQG+iS8dQ~@K#0tS3LZi*%kTrJW7)|
(kiP#3ZY}kJl&x|7-dmE*5mGnM9r4t%>N>=-)}xYH%uKn8jp5N9TJiGNpZt7-T`eoKWr1VGvQTSq*7yOj?iu;=j;m}P<CcHZ#?_~ET1pkkUnIG_$S4K;~#=S
sYhhp&oT?%o3ChM)m;&jeB>f4y)w+ZR>{uZGK8m+1ba@t8z58gGVY}^`VOrHhi#_8{>ovQB#8(-G!{5bz8ei)-$TFE=-hGaYl9Art2S7dJROeH8biw9{x(m!
Dq88@-xW{ufQkPQW%}Jqhz|Wo`R{lP{CnpKPSxK+{o8$mg%;@;k1n$xiHk#Pi>%Z<vmRWNlSRK@a78M?!5)fM7rd1GqZ0E{5tSs&ue(R|aP2ICubj~^!jD=~
PsgeuOi$yE`rmBC9%Kt7xnc+X7seiCbP2eT8ef|8-35-v%@l5^+rq(2E$UU97u-{|lhjLn!SDX{H_@;sHffeeIh4~u9+<f`&SHdq2Df46y*XtGxV9F1J_)NM
^`#6s|Nq{wSfTOy;vRQbocwoV>b()`y~yNi?gGQnvpmT>8Thr2PT5<}2qj8C?Ek7|;_(7m9&M#9LXPLA4eaGs_>{U#IEf!1B2p<(r4JJx!Lk$yZ#Wq5-njTN
8m^ahcT~TIp+BI$CxFEpGL6<9#i{|IYcH2sk-7nmPkihkivlR}Q!WPY`E^tN1OF>eF?L9nN#mRazRyuI?Q=H)!OhUS`)55+Tb3@>L-P!V1#bHZHEf}G-h6PB
)EW2o9Uy=G$qMC?&5v8dd*SmS#lh>WG*|+=hR?Z$<L2$A_FU4}c;<RA;Fq5^oDTZwvKE@cN_NU~;;1!V4Tt?q|0{>?(RU6r3?`VNn9d|5F$b*#C;eKE1zyIB
_<OxqgR7?%Z!M`5<WlXnDlguGvGOL7(RoL#j_H*Dk#a*+URW7Lo+|<vKRHu{0^~d0$x`nPVb7t%!qsmK_a_Y#v+nw6Xg_7+aYYbHgdUt5UEcVZML5CKX96io
YnO&?Gpvnfbv}OMh5?nVu$@>}n2Vkev)8$TCx2#TNrrMk6q*pe&}5GJj8CfCd%q_tvy7hH<;P7HGKVN)E0FP#(-O^k;r;>tiC=ph7n(~J<-4wngN~g~oA&xb
^ND=i8?r!1)i)YP=RIFJl)B}l<cJ%pi(QYijsK6R^N!~7|Nnm_5h1Ic8M3m;yxV*4z4zXG@0ppAGO|aCP*FxvNfDAHm4+0NBBSE>`keDS@9$sNf6sGW&+Bo&
-*5MGu7Nv3#iUhc3E1H<?&?)^g5e#<&Qrs7U~niZ;4f1|<p8JJ*Bc6`lb!d}B=Gj8Rai5rmK}z+y)-nboba_ifZe!59=k)Y-zO1rC%4IX?7}Gn^rVP+d>!`z
_54-R274I@kNX$~YuRF-hrPJKeS6#-N|~MBAnYIi?t#10@?fr49V>mI56bO#MOn1UXd9JO(>+h%Z9nGqe;KM^3U3$JB+Oxr?z1Bs%g!*F;k%wC9|B`@^Q}1b
i=bQY`@_@gkNTFVgMAsrct6IdbfzT>^X`1Xb?$_Gn&>TgnkIv913#aaDBGd;(608yAwxuX(X#D58igXBieLm9WAg9zUNIh1c$uW%Zgz2ia)8>drLq|YpAn_6
C0OGiA<uQ0)UmSPqo<2b3VR!R=F^{=LiPKwzlXOo2HGX6D9GKQY1W-b#T<xJJEmfqYMEGeDJn7}^2B?pr|i@1dT2^bAnR~6K|mq7``{;||8z<(JvTg>O}KvY
XLM};tgyNI$n&168<<olJsCO-@sslQ;Yn$GRBH!Hz0Gw%`c#45@@Z9UeijsEZ7_my``NsVT7BsB+nMB5x*|C^cMr042{~iAJ5{6x6EgFW7YxQAVRV(At8#|F
r02EVR}LuDbIO=bF~nWzTD_NF%`o}tZYT9)eLSK%5$5wu7H$bvj29gUIegic;aX%EuCR*@$IbeqQ8JnCT2=@&gXpcJIFq0u>`$LEc^?T9$5L`yV{qLmMECEy
ClUqiZVjEOMzP;Yk63pdh}FjRE6idLuua7y=~M`sr3B-mmNb;q3TPeb_r|^Xq|HGSKYVSKw{>ZYghQ-?&NcQ_$P~UUSyHdVwSUSh;;NH`UhjG}Jj4U43AMUj
Y00Q|eqSWR+5nlWI}eNpYw*D6m1FPzI@Bz%nEIs0;sgEb=NDxBG0Tv2mNxeSPCt@2kM~Z2Xktefhh-9`?({0xw58!(OFg}wpdT(?<S600;)=CVY5Bvan!wdZ
>!xzM5RC(A#HZiHV_c~y<2+jyg50!zSey<=J`0b=h{7=DUo*|6Bn5#_GFqVOkqN$zq&lyKw4=M8+_;q`4*yi>`5GVR!RYeJ<@g)%SdL#9rKu+D4~_D``lLVn
uK1Ie(}W^_qQ#o)OEJoy$UdVlaD=vOfl?R^AvcNV53yW~M2y;Y?(w_rI7>csX>+(541Z+4Gro_)QPTZ4HoK~@{qUCSeVIH2XSf(j*64#FE`Q;^jX$2fRnCi_
NXFobB6)4wExeK5lChZ_gO6a%(CHtM@bk-5mk$cY!`JhyHK94+vSg-YohpT7$W~};P$r6>Fy)OLiA6l$Prs1v8mL=(OgN<YV(E6i6wgKwZjLq((^MoOHHg2s
scHf)7Sm-0!$Al=$WRqjWrTr;Qnr4@zTmu^d*hLEB1ZIcn2xKaBJ1B~zwMM01Y-s)E$`MNE}%W{$%PCox*mD*>}L=@a(zj2(}}~u#D%5ii}fgUA*WAi4MV+=
u<_xQJg7;kA>1w>3TaOQ-*$R|MBZYH_E;ZIvFoVX?RCZR<+?C`+5or&m*##ZtV6zdyx3w!4=3!^{~A_Af&8ClZO5LgAPF~^GwhE;VSj;7cytIp55zR~vnAop
#@T(I6NyOAP&=G+uMt=F1wL#tDn!%)>K%@0eY|-^_o6J{1y}YAQE3OtBb<Li<<Ocp_RzYV-tCG5kwZ}3A8}hk9`rTrG65&`(q8oX%Hd1a2i=bo<#_Hzec{-4
9PBznzUJ+z0qcFbl(LEt{Ogr?7G~#--L<NU2TSFc>MJ`KsqBI{7vdA&erF=!R&?>XpZaJtC}--Su|f7JZmQThJ0wcgpyQ?q>Y{8}k89*0a*?loDmfc+s_I&j
%sNQ?<X{+}?1%S%E>B1#N1*;2+hLj}1tf8}Yvu6TBkq;DDHoS2HZ=3bldh^Dx1{gp8*yiRpzf4&;W9>a>g&sAKPy91_Hck;Kt66h<9yYX6N5g%NSmVT>Ima+
`JqzagBjkRKYlgoA@IY)u(0|kXlHCB_X`gMqno~UBq<Aw-<poymrRC8{+KZF%TTljr&ViB+n^xOTa|${3*)0D4%X887-2s<q&k!VZt48^&dn0Y|NgDNL==Kf
hP!S2TM>}??VNlm#U5qbQ=iFhn}W?DV7vK)C#J21<1;S%g3K>e)j81<zPkFP7hVOzBfag{V_7eRx&HkTXdjCakMLtZ6&zrCwpvx{vn}YjcKa!Y>cB!gc9!O)
I^?;VCq)FqAzHB&vtaFpL7`k`PCJYL@83VLv1|HX+Rz-WF6tC3<(@b%^Jcw{&|meB8Z4X=@q%kj*YO0KN!XzCW$o)YD9|`j%1#DBK5+abuZs(IP9LuDkaEOZ
vWJak#%|yeE!O#w{=e_e8^yI58lv^cJf9X65X@&Fzgy@8X36Na4Vm-c>~)i|V)KJ~@b@ZX!2qN^Oa9uF6%F@2Ex{tKcHlWVF8|QO377YIpRw_h!i(#S*-K`*
VBfALW%KmKTEZcsC5{}}p6;d_pixEq49~qd4HYmA`%n&Ux+BJ1w%LCv1?^c86l3xVP`UEdKR8$(7h3mRs3hd7sdGzc)U_0pif`qbrpO|@VR2`)UI3x-^v3^G
gTOUCJ?uVYf<r64Po!345v%aFba9p+{7v~f38Ci@UHC$4M$i%*315y9o%h9vZ&WJZlrS6)2E7q*lY#bi2`lNY6kKs4J@Dt30u1jN?PKf_hV@Shtt)T1a6C_C
T&al#)6>*7%XM-f@6_{t^70(6%BPX;Cn`qIt7gNeWd!^*I6*GDqJ)b_@BKAcybIn%TY*DThG_8T*kbc60lQSDx81c0<XAEurJfAK%N{oVtKZ`A@gvn9Iudm}
xbpS{OM?ndT)nft{vZ%5KW5T)2)S85u{ii-PzNSQv|kJ<$wK9xyer3PMMz(WF1WdV9!aD>cBK1kq5p1oqApzoYhyz1Wt7|@98A*mgpC_vjy&o0BAWPlRqEd?
lQ)81^FLYmz=lkTe@;KhVqlv*Dsb|)Bo0gEao^)Ng6W&7s84>Ou%R<!iYit{%VVvNSA$(J>fpWp&|Lzz{8FnQnsH$Kk>ubPJ1N-Qtn|?(@Xhk>kPWGzGTz7h
EO{Urhf@&)$|pbQpzhEXbNy+;{H<k+^>Rz#E#*Yk9}5#)>B!Wk3bI9_Na?y-vnb}4_Z<~IFMta>Ob1lI(ZMY;ysuzR1;e(lnl1jRL-9s`wDYyI7-_TP_LK_%
L(e<CHfm!yz4m%;bW9HJJrC(qOxeLhDVO!ATO9*Jy!YAkLeY8WWlX}ZIh1~i#wd6gp{bGRdY8W&czLzewm+T0xlC?%t{MgSY^4rdE>l8@y|{9JnJ2Cnp5Y=t
D+!J42fxL)!tw3RfoFq^I<QKhCYAl7h!4BoWJ*TL_~<CJ*Z!3Kf1K4?qw=Lp+yHJzU5v(K#F2h>qxsKY1qe`IW0$q#MZ{B@UyF{;=xZc*=m$Z6A3S$nCh<Va
S!w72y)QzD-V1a#sA0cICf`W9621*i%X-_3fvKxNB+ZZw%3r(L#eWLmOsOzWy>=na4c#>#qEJV}+Ou|VLvA!TF4IyRc7^=;&zUYi&Z2_5C@Gc13WfA+iZ@~|
p~A5_iD|(Ad@*frKD^_H>+;W&v60GnWK^J^>lcop@sO(7XIk+7yZ)MvK?Q!9)qACvBvF!>PqR<S6Rr&HkCbziU@_w1V=^0v0=i?1i@Q989$g~xFw_+9Y<V+D
70bcbl;;=G7KM}RvF4+nYhW<2BTzHrfwPa2;z}<n!v1k?Y-5T%LH{-`d#w49Ikp-QFlmcN8Fs~0N41e!YQ9#u;0yBl^QWyT!{L1P3ajg+1CBmOz9QM^gea<T
I)w+RD0n~>xUOJ|wM`{nTR{~tze@BvxT6FI+6va3hAdRQ(&IWJZ-B<(1M1W+eh3a^(kr_ZOX!)4DItyesLQ7NsCgm~p%RiLJ%0MQFHqobIHHSgow}1pi-gfQ
#(Y?vUJ0f&XIIFZeDTSjQ(bP`3Oh8l?t6rFaQAuht%PkGh^O^^?%=0`PDnuxGpz=U-+f}`9T0#H_sfzjJ|(c+dg893!h|^6k(d3FdMJ}!_^eGV0NS`(<8b{T
%>CQGD@yS9x}96&d<#zK7fmOb5KMqBnag@1wJ*Z;ZqEKSQpcu&mdiSYE+$53(+?GTV{T<dpgAQRTt=RoGtB|`=W3W6)ZmTTG_Lk_E=6pM$9eox7KPC<Jx5+b
k9;jRFSau1fd0%{(jq2p47&c<^&Ss}Hf`CH?OA6u=kpo<?w3P<T+jWNH|%k-Znyp$l_eIZt~Otrv4zXG54Q#W9EFAoP5;9vdH5f=yzhFvJ5H^+wap&j!S_}c
hi*G2R5!#riCj#D;TU&=U7ssnB_umK>8oS?^0B0%TiU2M{>ovNp#k&2?uVrQ5~y{!w#U^;0dMk2o_dR!qyJ0V{dOribYHn2^2y2%>nDbX6RFQ2ez<+<!Iv0#
FRmPwf9Ve@htAuTpVe^di}A2=g8@3-={i1bdV!kdtLx7f;qVhWf8UQ@2Jdeb<x`R>Anmj1p{ugzkxPAWs`9ck=#>jCYKC>8B|1+X7OIGoTLT_PWU4`Qj)%Nd
h#g6dZG#<>wx}umzIHkD3ak@kiHOatKubFI*X|t=uC;lKJ+VCjwSL<?;wB+nP2kaJ78XJag_D>si7Yg9>Pbk?6ru2~efe;hIW~u)oi<D?!R;x0{6>i-Qdt%s
yYX`)k?!*C4~_CTerPwjev1hwQ-0miDvyBT-HF{z8YQIsu-9pL?hOr%#h;ACrqHh4IyRF-2hSkicYLFwSSK43OYaN-iTNu^iX2wd3|&_849bQ5#qFb9$B!ZJ
dw_D4pDqSph^dX%Fk|MVtws3f<5>7$nxygfEMgYaA~mz@@z^rzlUAb*iU;^L*F)s+UOFmh`+v{L`6eFY%Vx$t(j!^S35Forwi_>LQ~+)A$9Ei_pJl>M`icHo
4Z+!NI^i>>8nBb6e$5li37+3C*o_rqF}|1K+JZj^_7?Y;9xc(uG-Kb1OXRlbT5atppio9hq_f(jpFC>Ym?eI=Iw9FxdzkK_G9Jj22}wV3hN2mr&Cf4nn7MXQ
$8Ayv@`tk4y{-r&$?T`vlLdQ}|DtqgvJyh|R1<4&x;Uoq8#B|%3*q0n(_Rc2O6b^L`<_?Bfv>9!ANck<fHnQ%D%W*cd=Sl)4Ndg<&%bSiC0@?WN5F%Htzluw
1>V_N0)kbNuxsbo{(D{sO;SCIN^!<uiY{m}pcVnO(<!@`Zdp)U5_Qu&@)(`(IldT1B;dz=`;;bC7CcRty%;k7zyAwP)qhz~Kvy0c4SkY4sC;D_#}~Cg>Y@1c
^CK(VozF3oWAw*OVHmX(#^CRL>4Vn^`_2(~-_x?l4`#`C-e<D~!E(EHB7|EK$zQCD=hd$whW)uPt$Z1zr9;nt;LQO4dHpWR*S653aP-wbuMckC%NJ~JnW5`V
ccl%FFP`goUc6Kxha;hqgJEo*$QT?t^TtREf!$plyna{E(!dr)-L8Q5H1u~wrTKAxWy4+nVhB8MrU>M}bVjiB1`)%oGAK%aUrOc_#q+Q2O@q(*Figql{IkR!
brbijM(lhEM@(Ilmx26^y&|U#Y#@Df<>=SQbUeHe-)%6W08_2egVk@Pz_NH%?P`z!dR$veA2^)Gp!u;UN-vcm@_Ua@X_+)Gq}x~1D8ypX`E&)}vMd7q{yEh@
R=}~kv66#Ap>V6@w_hamp10N4yFx-PqAoLN(c3El*#~+iesVdW=~X9!ienIS{zi3Gf)!*(-jK2X(#4pH_2{>}RGcGwvGM#}FoG{$ZWy;W2EBS#eEusZxV;S{
+J2(~jl9~J=~#JWS|@Wn8S(%^mh<Js7611hgH4aI3qtSo>Sh+7#nTw)%tCe*oE4(Jr(+q3LP4RgfeHL551F61`ZyLvypkG;cMU*xY5c?_u|8^siM+%Y6OraC
bU9tm2pP)rZA#kuXbIHJU^y#;Te|b4lqb(Z_I5|c(uNKm$*J&CYX=~v$NzI^xCw~omhU_L_Ql*0#@=66Hkg{%c)2&m1uaJmmzvI)BPDg@uVS?$_It;dXT*f!
SdFBVOQ0yc>I-(-uZTcx`^z&bN*0{H@Y&VwlM#;G|Jc)9s0hr^Docy-;nUvrk%K8om~XV)+zl{C9EA%t)loIn-j!Nipi;r?(-}vy6NEf2v6t!_3xHln*+1D<
d8jm=V>%-20@*`mQ-UI{I7-iCIK?7>HEa2DswgdN4NVJJjTl43mYs?`$rqK^-G85?R)K}MjM*{&P`vr<f66ah7qb*h!g2|Em<&77_svib$5)f5)2HqJ)2YfC
#^RtkTdaD#-kvFy!mHQwj4WObxUvxV&V5n{37Qo-jaea}44Tir`!f*C`vn`P2=`;2^uq-zE`0<%{;jpBrjJxz2N73aeHaT4pV|{EhPS5=aBD5{pkv@^Xy*r6
=t`ASx$#{DOV>ug{C6Gbx?l1(QWC}<s-dBAV;@AV%`J=!@uTzJPC?dPf8-oexp|@}9I_2^CsLB^u^RM~LPC@u5^VjM4AMGKuhmx-U@L`RqMBpHV`GTFx<>9>
uY|*SzyGRcD8j%`TBMst2L>H!Ogc;Yka5%fetE?N55k^$)CP;-$Fp+l=?^xrio4w*`Jfaf8$|;0NfjV#A$yc}I}T(4>WW)jVW@q)sM=<4f^g42f}0wCP}M0t
{O2z_WIflDYdoCc_whq)${uC3%XGLtNJzw7!krjz>I{4hG!3EN)xp68-yLNIcbwqfFZ<f6n&3CVXKufB0@eBZy-D)AC^k9tk*?nq6UMYg)K8P4ZLYw2>$N1B
Wsl5_TyjC}wGFXTc8TC>Du}ex(Z*fpSNor{#$h}`IKH7w4j&h89ey3+k2zt#?dSWHaLUU^Z7f9-x^lCWUUInv-W@NY=kdeH<dc&H)zh#LVM-O(cEUOLsU)pV
M)=n=aI`QO!Ybr**YIf}a6UQd`ITH7D%qs{oRJ(bUS_<!6y$=fCXE48X))}%eax08KNejbmwS5%e9z*p+_)s~4z-;&*31=u7`htH(YXdArSGBE-Fi#3&~HB4
8|eVe?(q5}`SFmjWq7SHl#9V1N^6y*iHLH`xKuP2i03(z)j8#wpnOOa`He~%s`43jBDVVYA$9mhho}$6SF}DINYICK?_;IzgiH8E{OIr(@gNKymkJDjs{<R#
-O<lKtPq^cEK(KVi=@NWG?zrpFzqknKUsPLOC{^-{(r?0N>Y3D%p+5{3BO!n?&X5K&CZey0~^7Q`Wv4d$OK2FrNm5L0Qf5p*6wH1#2<Gnr(YT-7*J>_|5&Yw
<CSF%CCBApFYI!*QP>!>nY@QvzZ=6D2gnPYxDjtTnAA0G1G^)9FK)zhf-CfZ!J}Sp*w%`S6`Doju*yWh#Y>8a&~<a&zuyUd`hH|uR0Mt7Gu$6{Pr|7~uioCg
VSse+Zq>*FeXM;sB~ET5hr8;D7KH=ONIt08Dna0=%{MB(kzulMbGb*INSO{&?@ZqPCj_u5t>bd4*9Ho{Jdc{<I<a(d<&f{NCw_!1bgVNTz$(L=&z+hq$Z2@v
NRlK&;B%f^&rV3A{H@%x%?@eo^E`WYTP6nurz<jyPx|7Cz3Y=7bt=%KRoW{N=z`LWD4rG?L0B$+wsKUFMf&gPtGe#|_;gQ#?!qZOoM>C!q~}(|jk03>OQWf{
^(*d9&NU~@jS=-)OS9tLi};-AVp04cidab(OCxY(;-%7kg1FicX57h>j8+oqCI2N>T&Pt~*e3A(_HA|ZrY3f1p08SCE@OlQ2YcbeRuL3aH;|QIGl61ejN%_^
7f?B>RZ3L?L&A!kk1j_-x8e7Aq1qYbSBO4u-7tZLu3=C~nl3Jh-93C;cuVFpgL2tcMiH{)-Vt$jnBe6L8x96*0jNtwo;&m{548;QO=h3@(bhhxaB{>5*+#c`
;<F7vEAZ{cz=AMh6?r9_6m@YV`pT5Ut{2P_=>BZZ6Z*udeeS{sJRp2&ZtG4q6|VD8wn@uaA(?pc?=L$^<jqOMD&LI-#RtQrZVe&Ay1LYXe-zN**-WDxt_Joc
;v}!vM(7o!_M5Kag2%7m^DX<dFgdMiZ^^0-|M$w-x%C?V_g#m(kCi@ta6@ne%V6XY1ElthRJ(mv1g~Lq0#mRMXyxp_|H%yp$<6ZUgfrq`&n2>sRlk8F`ssXv
xAGvE(NE@7Mc6+LC0cG}Y8ZK1=DuMJhJEv5&)RGf<igt>I$9#Zd-+iPg@{amp3I@+Yz9KKTK*;vm%?0F@!PpeahRndDIxikh<6JoI9dM&pd;fLMFMLYLcN||
m(TTwGJ}2RajGCNXVj1v-$;ajq2;-)%y0<PzSH(JPC<Kiy1!s>7J5`tDc24sApJZQ_tg>)>~dFDXA|aXr$@w!=aUVph@M3r5PbmKina{XyKz9!$AF!vHiXo9
JwEoq8*JqQr{l`wkf_f%J*^N1PLITWLo%^w3JxV54{*o#KE>V=YhOGp7PC~eiNt{;YcI1+QxIGq!Xfvx9+b7A@2?5OBG$m_wLps=rj)*_)RCs4`i{uN=lKM%
w5GkLu*t+fOCL@585@MI_zl=c#e#1ueD%u#SBTi%J6N-ngun)3mh<0I!5)z&KEIfN3&gb}&wbUfYBg>%Gn585lU&z#M*?u_d4S{}PH(s|m*nnCG5~RZE30{F
0v_cuJsTd*hJ*GVsyF9tap#9TH`8(|j?cB3pMM_<*L@EoWJ<C@<e)Cc)m?_`)|Qjv9BnY$uwMQ2yb9)#iP!pME@O6O>Ue`fFiu9zd#Bz`!rNH+7g7g8u*2ef
DQepUCth0pxY8Jl;U6{6AKWR!-0t5fSNdQut*Pl@%odbC^%Y5%JJBGrSD3Le1xM>u@6|Tl28(Um*@hxJIG?!D<tp9@E2C*n%Jq0;jJ9ZupR0h)PDUfAjsr#?
Zmd;(N(Y06!`)W|o()^IAG^g8j-Ig$%O}yHh@81??|3I1AIS6#iIig@thw&d=A3|et;D)Ax6AmmSt-?;oq@-DbW-{lqj8?~$!11e2uk^=IEEjDAbSztlDb16
Zt<^>Z@nD*$$pvU)rEkb!TuswL_RETOcc~g_+f7K?fo;;q1ZB#Sbvb|jMJko*X4f)!6Wl*YhF(}M*RY`^qhTgMy&F3(<@s@-k)Z=%@Pl@_YCx(bV|VV-I^t9
HW`xV&JMCk=OE}-p6cl7bXZw7u@K#B#e=A=#T2_#GzR5KYac9v;R4687wXCX>9q6qQHHsc2z<Rv(Y94wil)UI+cK5$IKo&aN12%cO^e8PpKka<aIeX@2=ztW
?<Q`PPIgCW>E)~i(r|od2+?v;zJL%XjWm_9ay$%IxNG1Si@1CDm($|xaONp@`&m<8%<2oZ&icecl6QY%twkFegraSo3IZ{eU%9Nb69TT%&hrztz6kP4ubB1H
1JR4HzJ9GT+<)(w>G-=0z1fq&eCWi<C5HP`>XBfkJy3knIs?U`2Ih3ab~vYJ$6$TS7N;E}W4<Q2AzJR?!r0tJ%xZ@2wLX^(k<9V??1}O4_P#U8Y3q-jHc~0^
fF#`i>@c4CBLHc{VjHnv!vK{(iJb-hXkX0b!@fM|e+l-EV~>D^`N&(<&|HkLI33jqF+iRI<F@(1aB$a9QhoCCfXjIGcH?vs{7x!9-zBVfRp`35^5aC<_=a83
DGfkS-McgN!MVtw4Xsc6kOKb%-`@}YTrl}BIXjr46?L&jRAS5F_|lP0(O(>g?ka7cu;v<Yy*X`pN-_<~WbI5geGZ5c*k+KmPeSmd<<=qX76it2j`aGtL%CS6
=M+ma?rK>cZJabkbor~+Q(d7LDLb=E<(PsY#`2!}y|xJSDzzyhu0URxFVj$%Is945uO8b|jEb;xVX^SUc*--sqAvlkSidk=t!xYSBVE;2uHn#iS+pMEwMCpT
l=$g$kbFIS>h_nb=-|k7jLZ%Oh0cpW!=yBjH(R}Pl}mu)-QeJQVmtI^oqMaJ;fxTyYgM^^O)$%LbiP%Y2LFZ?w-g&69CbeY>U&Za+6ro4HJ{GFn3wRC@A*Lp
XnQ<$D8vKDjSt<q;+6+$>g46pNq?;D?K(nrAPzgX?v@;j3d24J2cwDvUl`5S_`1kNLZ-2rZhCJ3{)YQ};~2SuiB8pDqV;+hHrMCetKtIPr(|_|55>d%%E_LE
+&VN2-eb{E(uJGnwcWEGIoL9v)kKs(R)_dj%u{^e=<IEoO`ZvxOP7_NsM^7QfB6;85f`X8e$_q}9R)J0m`6`1;_yv(x$m5AC7#3yslK{HSfAr!5UsltzFhZi
E+fnxdA}J+Vs<<Z>~Hm+9CwA2)@HW#(Q3GMx^BjMhQP^ZBFN;WBM#1%=D6&S!_@u9hKWH{5P!iwHZfui_MMW|l#*CzS)48N^DRJ%mgCy?_h1-I<mpJq1|cMi
eN&O%5lQSHZYt0PLc93Z(EBb+Si8KV|27hY_Ppfr?j!kF`zqgD`!p7B6cT^0lI3E<Hn6broIhyaZcuCxd|{-B*sq!*2%mQ!bXpwsM*uMy`7`rE@QZmah^fTG
ZLij{{M$%ym>F&d+S-DF-&rxaF$!f<oekW{=}=_t+_y#)0*ZH!Q<&)kq0(k<h`B)QexWOlve1K|#Qin*pWcXPdK@_HQimg)+KRZU44p%w{pzKu5W(2mt2cJw
9Ch)()a*yNUK%Hti@{hIdZER8)fPpLl&eNDZkXRA@h8VL3pC62SI+;AM@M*S=E;juSkdUyau19F^||b3duLk^-O;LWK9-GFB;PC!Fy~^y$ne`qb6pq~H)UOx
jfcN<3N4@41?<KvpE9<qzzylsDg2BXupNIILu_M>CkOZ6+}9e0ljJ0<uk{JKX3H!urFVd3=5^s0%X)Y(elD`?q$kQ3)BTr4obY}|M*2ohK3a_~WJbgXW7Tit
x!MS!uMI70&Hu2+Gr!H4<|SW{+DT74Gsoa{8s$@6<2Z~S8+2;Oh=il{AMW7-Ur@NLGX_U_WA&Gp25Dayq(onRYRE3YxQog$lHa!f_uNNZcbMt_{_hl->_Eeq
gXh)R*E=Kp@pQoKczCr3=4M`S8`@p~oi1OWjAjO+gsax>Do4YC^x*z$`+Trcdu;csemVHJRw!g;0+10Iqx5>v3#C?F39|JmxF+}WzR;Z{Eb(ZMrLM+dnWJ<3
yR9GgJQR_b@rgiD$m`v0r2t%MZZ(&3^Fb`_iREifvG^0*cI`XC?`y6eD7??&NyrPy-a>y*e7|tGOt~i#UNbTo0?l4XJ)k-9btDGkhIPz!-C;QRI%nvxLpZ*Y
>F0#H8DYnn29Km{p)7y-i-w*O(%)7~w@gN0;#IHYe76s3e-4XLbOqwwr%M@K!k!SGlbRX5;{-BAv)_8I31E1lG4|=39dvwJ+Lf<6A^!GEhf$*%in`55=zpi9
ohI9+-8mXoYs)m=dF=!a)2Mouk^)WjckfP|kcHJ9i^I9ZkziO}<rw=I3&W~x8R9n?7`YI$Vr*`WzVYim$1jCJdp%vIN5U1${q{FLhWKDm;TX})u~^8&Nj(t0
myX1<Me3$O{P13sB<7s*#b2-TGR0r!=#p>B{zhsA!KG8K#~+I#>Rs4}-vpmxt~hrpmNyvbTt??e_{|Z<)gW=`NE8y=bbb?&`@-e=iu^II2>hBRT_<94L9Vn?
_k}w#sGm8(c+D*sM4y)QpDM)TQz*m9v8TRBapQ_$-3rFGXP>6Y`#@x^*+#jYGQiRJzCV(heppI+apu<POjMljOJ}`ji3L(p`?=HJ(D^*}^F~4x(kjYirYD1O
RlwET;2uHW0llPs4qAA=*rxwh=Q0EZo^n0t_e6lgQ=M)8Bs?bZSs)>nL*CU}$|?(Ph-x`g_ncS{W)h>k3W1?u7$P~<qvrrqW)rvM))*8Xsv^H<ABaf5mEJfJ
Pxy^4x3m)YxVKQZ*~C2rz9Z#7RqbOz(x+_F<{E`^z32SOHNL2*vTSU7;Rk-=-+eFd1S0m+;58SQ0JQd$KM+3{2j9bO8M-!(=qxb|-y>~<bBjdb-YyP!`2E<n
vyeTGYDe4~coL2du`c^LHY14s)a}jD^+h>Ng~soJVB9;jNOn0b7QBtx^Yo12=#o5b#WoX7=qawL26^$wXMZ4XWfP4o(r;<5i_Tc)9JdT&4aRRXnHSsUF8}GY
oSNJ;Ma2W%Q=;4($^Mw@-F-W;>J5n^5q1rx6m+}pe4He9grm5@!<y1ya0lPoVa;^F`|<qG?)-5m^3$ijJ&=UMdwHMhH>Jb#ltsd|xB&FmSMZ3(nBapvjis)c
F<uz%3@H$D!NHLx+w^rXl9g6`Ps~SPQ^W4hQAJP4A2KaEw-}1Vx3l;&sDOvs{Ri+O6fYD;$31D&u+!vvl#eJ6w=0Uszg|kj4cpAQQ{y3+HeVqBF75%<X2<=?
#cp7dFROfeFbaBIiHsHnDRA+evr*1YhW*XSf{xh?$VavuFZ`Vb*RCwRO~XWVygOO(E7%&{@A@Z>4g}(@I{Ek5Q!Y3pq%nU?GzQ<;TTN2*(lF54>s3<e1O^Y8
kFuH}AlhCX9e#8fn-V8iY<~LS+TXJ=e@F<|zcHwNvC0o`JgP|W$q!#?LmU&yVo^y|RT}<{kQ?Id?PXj!c(irmt+|vh;%@QXOs8!Css8>tQ?Dp^EH#`@-+vXG
uF+w1%-JweSR{%c%Lm_MT8_VWg0R(Z>+&)<53{#kPiR(EVej=ekM|p)SZDe}XLKtGmFs8pj};qWA6;_lezO?Vol^FzTFXUo56|{=qaKvc)st-R4~Cu2j&T9I
EA*}p8*Ib|<9N}}&{^v^^p6^>q;dpeoip}{lDaQgR{E`um?wg+mNoX9oIBhf8yx)gl)$Ac<xUC5qH%2G8Ch^wIHm&rIdzX`Bdl6;I6uY@eB^s_r7Im!Rx~UW
v+Rx_8M_qL-wB{P*TT(c6oOZ}9JPf!KImOH)g{>=a5!b;#iPF~KvJ$o=C@Re$T6pCfs7j<_aP1J;7Y``ORSlnTGGJJyPj1U=79!(1G3tHILsU}qo_-&!1Jtx
l+)7jn3eHzt&9xE`Jnb-`=^EYA)fNlYSj&Usr>@m2z<!2`LEcqHyl2r*2Uk3Q$U>77j@}}HO^dL^rXy>1e;Z8;K}&_FuyT;mN4XxAg-6XH+_nsQ+LT~c0COK
d#PWA1%$)w9kEX8`3qPUHq3c?F&=UDpYP;fbVfzqN{6p(5K@ojONomoL$_n##Fk_*#4arAEhxt!eQ(F0M5+UX1-_9*-p|ITuZtQkR8A0YaB0~it_mg1UnO*_
nYj0sZJ+zcbX<+dxySJ1K0&Y7mut+kV5&O3VMm>Z0z)S$X@y9rUu=4%RPKUL8`_N4!49ZmGXLDMW`_YOp{M5veECoF(xutO7?3HD4i<b=$25=rzQ{^B2pg7D
o!p6m_9C^=oN5k?FQrgi|8o^g7t1VSTkJuZG5h;>p8;0Fd~RNM%K*>kCr703`y$i$!ea%M3>@2DrGL=r3fc0w$2PBA(ZF}J&-bu9RO>tKW#nVg|I^^=rCK-S
2<irQB*fy{j;hE1-p^Oup390VyNADt?i1~IgYfQP?*~6hCG_{Vy|eq^i{jUgOu@c}@H-QC$aPmA&(|jg^hiUYqDuaqyv-T(kqqW3A>J5Pdh}I^J_&|Pao3vn
G(x;Novl*B7f~;D_`MD6vFoXN#qX#O)XH=jzie6~&w5vIOs){MVN(-cG<>)uRC)LPqj+@Ji#$0%mj{&T<|#M&p~lBi!{>G$I44D;72~wf_Hg8tW~eiAc(k_e
o(+YTrKCFBZ4W$W9hW>z@UhHq`ImfWoN?fKt9DAg5h%s-Ulg4PKpWe!KJA4>j9z)lxz{cQk4}z}XrD92ANz)^ryJpjl>T|5i;(Xm%pNR%GyISuXjF1Hzz1xd
I<n5g;V2w`HO`l529jB?>B9pl;MF3!%P5tC{XXAbNU6*H=ii}1OaTj=fgo;a>e)~t@aSN}-iJ>J{kvD2<j~(73=)MZ&)@P#!zIDIBas(T8FxMJ+pPdR6!=RK
Y-<XWnMbbNL_r{*%;=eY?FH@G`&QOt#wg<XV!cgkht*;2*yd}^NM(-pCSP{MGv2DYM`2QEq1R3OPH6@I?U?+x51p|3*!Y=ceHwDpzNfxq%*DmH!J^h3U3~R1
$r2#U9jodc9cBe%SZ|-Vj3;pfd4dng*d#xMP8R7GwrQXMwYe`QbRl%-7csl6AFk~mS0pZBz(n7X^B#UdIK-DTBdte_xPm7Y&dk;*+H!FCns18(DJ-vchYZ0j
s(MZTmM`KhW*#h2N#Wj7vmbN2CLVWHh2B|E#gW%e)hwTEF<-R&Ftd^yRu9!<JC5t)T}yX}jg0~x57EgB2q>YhBhOGbQU(jD)vII%=OAH}&7P&8iVc=Z`&9;0
u&tHexk0Q1y8)B&_GKklcFh>QCGd_n|GTEq5I*Sl@ejCs)q}Hy^i_HvTU>W6EhJqqhliW|!cI5|t_FePykZmv?~n@l<;x;3rjID!UkBA;T6N!q58-mR+?dKo
Nf->dNjBV6M2;PK*xhbr&@D$Nl^PnrB>Tnm<4`SRJtY2aAg%)K^qF76Z`9FUGq1-+*l(#uC&#%DM$kO65Z%OThs*`#j(dXV5U9p1s`bDW4W~Ul8CpGXIqW)I
RIL#@@QTb@TMBfcpGWF`%fi2>#CRxB38^Au9;&^zka_&4$tG3;KOeUBntjp&>RX+*YIL#D_>Icz!x5b9PmDjWX@J@z7vB7u;(!a0y#LQlJ(vdjjYL<|VwC%(
oP~uv_Sro<!$l&97|Eua(iX-DuUT!^=##;qpw_>3HCgCzZ~FZDEdgJvMxIMwbkWRmQYFTU6;)r~SOo0o!JLU>_48X@d^dcc&rO(z(XBToT(*_?Q~1kFFk2iC
i#7&V!ZJY9X=Ee3>xeDpUkx)Ak;tYBn~A(A4HC<EPX;P^Xm3^;&ld9I#KO?oUV<N~+YX!FD3-$V^+k;tt&`xB)p=f_uMex`fbt)`9{BV7v4uyqG%oSoHSY<q
fD_Z-*EW3`sAepwB`0&oJ}>n{6QMfzUd!o9vg&~cd&>W*H3vh>-_W~XN9sSFI#B#>Tg!@u_*?gzj5{81H+a-Vm8SwC(J}Amhtv>PXL_f?PnN)87w(a_Dx&!F
lMzD(H;ioswVqe2#NC|=#eWYnU_(Lc804>r$@T5*54m>mq`sE&tMCkH_DIAFw%UM9j^${>8+{Bd+fVRa)WHo+O2;ZLW29^u-8&TGijjQ}b|QY*!D(vCJSB}4
S4fUGzJ)C^dd7OH?mFO|HZwC_rY!>ac(2IPsH4?<`P{*)S_tgAZ8WKD0=0UJpPbQNsG!MxrM@);jw7ncfj-JGvU}83D;$o|+&6v+<?dMO*0_<?tpxS3`(GS?
I3oI6NisgmBWN{hcxR5#%U;?A7U*d~q3pxIMQbA{9pbt+72pd+k6*78u36w^Y{cF5{ANr<XgP2GB=AFhd$#Rc7sA|RJhf$Wg_l61%g9%4w9mY#>GV{<re_PA
p@9;-j`PeHsTxBg>WMOEhX*``Y@&X|Rv~sJn*YclSHxHw=!R{{AnVt=!y#uZ@j;|_PLbXNKlP$2n0x*3#bD52=KKvzIgQAd{vzbEcayV0l>?}KXzGOKopI&G
?m>Y{FFaV>mpRYDg${!|DqEikeI&^Fd5fzzo)8l;YBjN;OVD+qCBqupo~=IT;v{kS_xNGnP(2vvxUAQ848w0SB|LOrGIH}vUWYup3Mq+*^S)wbP$r42d)69^
b7n^rOd<&RJ~}GLmFR+fbEd-es#=5`+mw>_w#ToZ1-Em$^ibULhcf!I0J>B%GiSy1z^}u4{3zi(n_J1q|D863T1AhU#cwD4-t1GVV7A4lA>AD%6;&91WeZ%S
3q!`)2r=Q$@$fdjPdB<6j9EULgxt*<gjy{OF7uk<OH887^Avp?4R4=IQqsW@mT&JZqg>$NsW4>vE(8(R#qH1N*&}?9-iD>kB+gnB|BO`+Mc@I2gJf#9FnAV~
IhtmL`2@*<xX1oTW?&Qe$!-bKNd2AXc23YP){$L5auv}G15w;?gaYY$SqV!y_!NKrIJZ-XZIgfHKR9zim#0`e=dKQ_VCjA9H&t*e`~DG+N@u)rP-M7fG7IVc
TZ^uYzF_rbypbpA3*QUV%tpBy@cK>_;ay{Zi$?6Ve~%hsdXzk6I_4Av-3G?r5&Fw+7^i8XqZNJ}J3Hj&IfTfn=_FbUd)(_Y44NiM#Qto7SE40Ra9$~PYA$s~
_w{`$AB$|!@iVZB<FFp4NXx9x?A1e`fpN;aDNAHXTeJms9fV5WI|i@E9AJ|<xs$DL0LI8qR7``_NJ|s=V3DYRKN){1hz{z*?NvC7)CUoadcJZR{hW<L9lL#<
k=pPmHk`X^OW^f8JXy<>Z4eeYH<hYvhqn2z^Rw0}pe@r)q<9gDt>EsVc`8EwZT!8>_^gZ2D<Xf2edR&c;A{8(P<>SLEoiPYIfL)T$eWcL44AiNuj0C*goqDO
v8PS>aocq8t*QE1luwGAL=4HGXkt{nXFopz&Q;e7k=VhOFKkqe*$xx$3&NQyoRHaf??^o<H*!um9X_|H1b(waB$=+J@I2mdn32*Fp|M6Lk(&D0*VOJ<?8c9R
Q&zLl=}tJJa-Ub6#Rjro*0tvt6_9;+ghGwj8@qe<*QGq?h3$80;v79yum-;6eje<H<n9Nx{_n+*%=nqOaPcZWf0EyD6Ac9Y&~|s_KXFLd?&&j*)WRTrUr%vB
1o-U+W=3QbkZx*K_##gkhBMDBn^R=a;Bn#5$u%peR67YqbDMx~4-bFGurOxX%zuYSc;cD5_lJ!FL5N@Wr%>+HhxpC9>%T8ZV6DekNFtx=KmXp|z5QA<f{<(N
Jv~e8><HoQZ8Vcs0sGgc4xTH|(ULhU#S`Fy(Ueox>`x7$7)jB$FVzsF{?zVM<TBtpJ9Acf$qMYIn~T-Q8FBO5vhoz6r_r?dnBN|k#mbUw*vn~s1V}%t@=er2
q%L)s*F8Ik)ysya*Vw}FwYh({Wf4M@)+$A6ebIixHj|gp3YDyqMd5t5_|drDHJf7tv9u~P;}8#+J#&o~vhRQq*PgC}GnQaKta?3L4B%U~)v4Fg1*?s!fv<fq
Dpjr=`WPvQARfy%M00%j<wiL{WlM`Kti(kPi$UN<u*a20<}e+$j^BA8i4!b!6DJq=@mJXKvTv3;w%?=+98tM|+wLD$-QP;#LD8!G&)ZxO-5ipVYZif3ov@|g
CvSvBXV9Kyb_M$q6@OKUBn}+No6y-K0OsP<V~>->AX9o%F34UG1H4nlF9<wfHnM#9;x!K3=lran#?OzaWvSWJ4QV*inp>68sDa<V_ofjsfve1)f4Fr<7=dh$
W6Eb_U_T>r^-P8^;@URN-}Gq1RO&Zh3#%hSqdt2uW_aUHhGMbLQ+dLD`FGy>n-gMWDS3k<g<(0BuyMmi4U+K<O%?+>2%6cK{cP75qfrkn&9_u=&25XCPnri#
%WTWlu|g0CSeL)8D}r2}Jp1MXHAGx=CNaCCjGB`(rJHvYaJfMGz<UBWQE1UPwea%8??GC}X1pJIG+i~Mwsaw#E@l(Grhx{leF+&kg2>hPd*ya10zDU-U7n9A
K;$z6$r7C+-qa1;yHL-GS`y6)?>i#Uy81csW{?14{#<m7sgQ<9T3l%WzX)Vrj!ea5%47e@+jUlI^7!U-rzd_$7qX8@mv0x!VQ|8fs-uS=#hejMeuP|qaQbt4
FrOYI1j+LXxJ5xh+NH-q@Pn5Z%ydQ_8Sz5BrrFHg8IdA9+NW34;dAVu7OM;$s4APLtXB=7DIQVtBiarRCA{gTq7_gaL!tGrh81fITg%ruHSv@r>O#XkMeO;u
!!)oY0>_yCp5+lm4AoeC&)X751g)&9|4BLo6>$<3zg33V&Wv%Eh%8<#j(d^>>%o<ijD>4N8j{jiv<AYh(8u|HoMl!K+2fB4XO{&);mD`ZCZUJuz3Bm89;*DO
(=Eni$pg2<p=|zI@84ZHa9#Rl^6rTsii;-r7sLhe_VTNfQq}5crsRFOc1{+r{w}k1+jAp){Bmg%tuh=>@UKLk)&`d!|1o1F71YRZ<!kzgfN|-0A3q-xB-OLN
jb;eKB4o=fL)Hg7hWo{W!ez1dMv5ne_8EvKl>74h)x#Sj$JZD3F~cpNzm?SkAj-*nQT50d^~wt(@}71WolzIC{wf3I7kn2F&x(QP-s(}VN1pgnyX^SnumjkI
rrv7@$>Nir%X{NVQCK!|*wNgRz?82g(ad{mP+CThdwUpS%dj*wtw#jwE>F)eY#HN0h13&=plD1!HtP^5w?)n6hD(J<-EfuIa%3&i7bH5Q^PFKONKme>ruEea
gUpjBOg>VmwyX8j8mh-~@-|tMxeiKGE9V#b%+TKa*N-ON2Quo+6sf^VD6|Wyms2so%Lektv@bg$!aKUJj#e8LT)P>C_tZh1PF}dxqK1&iNe*{@2Z3ZTw~t?(
AMGoaQY8h>kWipX)pztqz&FeB2`eq=oT<JXSMC5Z4!&6q4jEh@*&4`{B%D`oWujuMFdnHK{}Av<5X>vVik!zRU?TeL!@zM<bk`|uGf=i7Z_iP=SA_m!WfGR+
Tf+=c96B4)Err+AXCHS+OW{ubR$RWn7W8t$XLxeBK*MllUo@8;Oe-x&`DnDzJZ)K_Di98rw`0E-NZoKqzy9EBDt%PnWjr&g;E2Gj)1x2h9WYri_m#9VA4E+e
4h&Af(PKX^+g5mBd-A~stHgaMojTA~u@s2NY>ml(>UZ#PddpSWR2wpy7oRWv)koi!*kKZ{2wXX{hdpK74%#85B{buDxYhXP=46N$7PLgJt{m6J@G9HSqe3Du
S`Fkpw5yE!Pwo9CbCOYztLxW&$^woebms&Gc%aR_YP4J|ii7vo<OX6j;at+3OzI<wGh^p@ADhUcLjJ&2t43Q)E{o_qb#_B5%c7Hum>%Q~H9I;Es6kjotzPzv
D!c;s6pK@7qeSfZ)ZcYE+{@8f>6p*}H;-TSY>^abN(>*=ys?Ixjb_HOrx29HxXRh?#vqbZ#$Ro31V|X!9=1)(p_f*G)ibRM@&+T<vUo4y&vPsBjs4}=<T6aN
{G<t$qTYVtm$JAnxvlm*gdc_<jQ5t?x?)MJ+^~VbPmD)DF%MSB!>ux+Mj+M=wli~SkCZte+bgm)W2A+l`0vX-b%yXNR<zmV5QS2w)7f*ENU=ym<>l;n7(w4Q
h&+AxkSfaAznG>0n^Or{pZdg+*z-qaj4B+d$>B$ZDFb1VzYrXlEeENjIuAW*9lRO!bh+zb2zfOXu8_l=prvmRj2z&=84@{5QzucB1RRrBIHiZqOuTLTVgi@v
;uEhcOu;ge%G&mj8L_*0du=q-afbrC$+m$=;d*Jep|giD2Wmy<#huWb%5=;$lpHk-Hub|a9N>I+iDv7O99})TQcG-h0`s+}^soQo0J%+dp6H}Jre5a%>`B(e
eQsKpEnQ|9auv~PW@+HM68F=aJWSXWjoa6`VTgUdMcs)PWboq?X(_+dvW&ZJ--y_<7n~k7No82-V8uJ>LHTJeSbC10DLG(|+Tg?wSsco^)Baj5klqwGOK-66
X(sgHWwu2+E*{t$l`T_Ksw1=JxWuxn1tRWQ#0&*0V!>J=GVF;GK2VxuerplN0Ovx>%R)(%?wUSaeJg?z%Go8BU&auB8B>zqE{E65G{prwX80+VZ1J&=6Sp4y
;M!AbgqxSh5@?Lo!MIoI;bep(A_fv3?J^PO%&D`)_R#(Rdynb)C#C>rf2j3ey)SL+i-0q0w+4l@@HdT~uUS|M=jku*&#>}E>j>v{$t5|Aoh)t><uAi%OB&nQ
U@&Mt&mK6^<AHxAG749vWD&Yg(Q<7&6mvR@LKU74pyCuAvG%gZsl#FNH`@bnh*8pNDI*25-qqFKnmLdt8-K0JTLQk)Gc^^|$>=)#h&@3vgpm73pOF{GLU}_z
M~u|~0dLfGZxecnliA%1(Vbbay0O@_O#cFZ=6d-UsGJdaZ%p5&JRC3g{Nijc_k`k_9B&h!J@#ubdj)N%;BtW6cMCxWv^fa<xVuTng9T|Kg5_aqadzmshcj;Y
coxm_nczuOqjR^d4@Qpjh0crkVvv_&*|p6ISD$RFT+-3U@8W|E84n3Pz2CMqAtV87Z^S7CM!FIHJ^Yo3k}1TKmwEoF*g>Y~B;_h07uLU*_<Ixj(cxpF(Z_h5
Q1&)d=JB*8&is6ysekG`{MOq(aOwJEcApm2>Z}nIQfV6XOfNuvQCTB5-v9p<b>8t*{{R1PpdkqvNhnEFR=sT=WRvW@_vYZ(>)3nmk(rT^21=-uiqbG5k)n)b
B`HdN=l9R&`|r8k&N<ihdcB^H=ly=WuGe*eLg53&%V%EbdFJvn+f5tSzP{LP^}q+tygX+Ect|K?TS;i>$wx`%&4qHZcc`fZX&p2#gW=xC)l3tf_-}Jak@(mj
x*DgYpEN~)_@(I%e{=wBiVw9@5bY8Fx%``OqczS9uL>2Ko1?wFF?g`u0l8nJySfB|@r;HtWZOj#+aE0r_9}Rzj#qJ(qQDT!lpGZD<GMI=(Zq#g#0QhY4bCsh
^zrnD0Ixz+3LY!OKG!w02A76mTFy=zOsr(E$QFbnGG9)&zR(N#t)iXmmiMu}kWlYtYmC67DY-W<xnjhr=>~6}C$@`ihJv?k@!Mig*NbUKsGb)(7k)bl#p<fB
eg4{ER+;hqPIgCdl`Sg`P?<qEO|>}roFjfW$v$_D_Q$hb@dxD3=wMsu8})FA7HYiI3a|0WV*hgXgE1NpNOEx=Tr%|pU3EG~^JxnxStWS6`uifEg1N|I!w(;~
KfQA#|NoZ)?)L?}stC_i4~gaTK*;TD!B1m@;G@4BTzSR@R}A|<xeF4Jv8hp<NM8%*7xSsSN377$O=QxG`~UA{Obm=OF-OeLTQ=`}5<%d<<!~t47SE6TIpy6F
3})+Um4dkt9F#3{W8t*N?mvsUB6If8q&$7HFWmlrb!rkYkalms6KZ7QCU^cIK<dxW(H#>uIDa%hl~3OUzo~t}f71as`Oovp=lOtBv4|RHFTu|;=23xn1YUWh
TEETnhC8LlLB<{;>bI76FFV-b_!EimPxF;gJuJ*_l<$tdEA(NucIhaQ{=P3r&k4dtAL6Nflwf9+*(n+0isQ0rEviMv$oR)4XY<|_vu#XSjWP-Fm8%)JyvGMS
#Fbr_&$z(bQPf=Jl@^AAZ)WJUWI>B)OZfZLALZL!0_?&D=qj99Tp2NkWcPK~Bo`YTbxVKohTjX~hf_@~fBB%8L+GH>Z(BHVNG%yGI-~cLb;{nsDwtBon_hKH
M69)ij%dsOe?FI6{ZrN$B$U1JnEB=eM#-<^Aw9M@^S3~as?`n$&&^Vd9Q6g2Dr?4!tQ}}t_eZ9vhM{M2O+~>ohMX(*T;bD}SnhjvERL%g_Q}Z}uB0rSP)e@-
c!>bXbv1SRi>}a_Orsvow#B`TL$N2HB|ypEo%_=_vj4lkY<7O{h10sj!c;@Spb{4w9-8%p&>eT00|!a?G`k}9VayuklGINg$`CN$k`tz_rH?1R@}=dU?4kZ-
lj%#JJ6O-y6WnM0@yE?c=2og1V#|&u9p2vq*N!I#n0pjJqf*Iks2&JsS%t@+9$ACpJaft=4=WH#%#&HPwUKa~!E=Ds2f3`f1ao)k!s7PlKXe9JxUF@6`r>c|
^d0_YwVE}etxTb9V%!S}d=Z!FYX@-F!XL?%uAp~YY;O~@gPYoswnG!a*wwT5qv~Wabatw2+)DMwyS)UH(;W|>S;P^4@R>bw3P!dXOC1r=QBR;=@q(`>?Lb**
4170k2~U6Yg3nzwjmG{~M0@<&e^JQ?+%mHlWV|$S-SW5ad#eyA*tA}6<w=2w+NAXUC?f7^ngm|*QAQD?MBMSga4;V0+$Bt2hxK8`GoR1eAi8h2$Fu<1|I`H)
9tjaKV?4lBy%LHO&U-`L_mgl@etV{V%?FAjgafoT-bnM2VZR`1gy#&;+DfOKFzNjFS$<YEcGfc<THUaMW5v#Z?`u&QPqG>wp4G;6!y}eSBS9FoHZM<l8-Oni
cO`}`<8h<+4#DxL7fNbMK2JnHL_}~c?PXDKl-I_zvrZ8p`too#)pG}2ePmcMDsO?#5rZqGyaaqsc@oU9!xe51bULc5qY;_b!hYW;1>;Va7cTT!Lpaq!*KVj5
Q(rny*#>9AO}I(a>~J|GS2y~a4xdKW@urOFKrsv--@7yNr4=@V6b^5yTch>DHNSVYF39>RO7Z>meI)G`eIcXb3GYX}vr^%=(KOrC*fU%UJJ0Gn6jLtXSxs=O
IAMy|jgNBYZ8RZN%+2wr^g3?Q?9TimPWDZYyAvJx1Pn%3A83vuVQZvH_{WGa7P1<mhd+@yfH=SV&pZk3Hq@+hx#z)p@N0yfS{B)7#zLBdG(qgk3_N|~7K-##
pX8o01j8$~6EuSIST;QVJahgM_y?*@HsYON!BigfpS%zDlxwX&_-qI|TX(-G&jg53{rTIzR|g+uE6q|axZ$WK)$9~+2QEi=KWF%T9G)`^y-hcxK-c{2u?ce|
HmP$i(>_(l6QjbT{@V#)Q!!alT9(I%YwGm=Hb*?_c%NxW_WmlG65TojZPc$T|M>Y;4<TBu2RdG~;@IhZT$lWf(8_h>)_$c3^lfN2x$ReiV&Fl0{x{aRN&U{B
i%tiZT^@f7w06Shq2ITxb-N*NY<EJl%>g-{0jm<XozZ0Y??TykJJeFWC>wZhi~zPh4NhLhIMer3(&DKFNVk)i2A|peFTW)pg*_118;G-<cSPvek}!3%QzOw)
8~m{{SF_IPK<$Sq^_Z_4_}bY&A3o>^{ZUo9jHW1bkF7?((;=V#rC6Ujp9p+3&YYC8Qbp*0j5PXOj;NP2ss9m@2QTjqU9YS_92EJ&CH{%rKLy;%B}pz&_g7eV
aF4`A<AdKlO6=js7#6%~<&T}yJ2Q6M$D{0s={aXlBEB(Mko>;fK#{l4VdEK3Jh>?ReuN?(bao5F+$p9wFsHaO6yb|C@mqCAo7|9kEH6vQB>_%f4t0CkYNFz7
@5_wC+W1rx-2SCF0!KwM=tanVy4)zYS8LG&%iiY6V@^KMp`oVqo6|?>kZingsxLOq^y?{nwL{LN@<lhsQZP!*QQECpVs-B;)-7jgs9PCcU5YV9T&|jQYa$T`
N7=>8Z8KobAfxOj@dAv#!qu!(>Zs`n8&nwd#^qyFV+x<0@FrBUeN55|!X*RDDq$k1{!x%lb=4lXWGmLQ*=@i`<V|tXu}7T4FK;JZFDQylYfkLE1G5DUXDJr}
oIk0Ztw3H+CQFx24<{KSZ(pE${Y@i~y41S%Kas^1xoaGQXS`t|_$~E|VhC*f`wJTzRpCAOZ1Q-22r{2&jcV{D!zW|yP_CvwHoi!WiyZOCTT+dF@su}irb`97
my`QYW!Ezf@o21{Iev2Mmp4A1Tg^(gB;j(gaB*a(E7*Ri=&UR#;NOoAomFlD5NBSs?5BvrDcX{Icb*bK>Aru2CEgjA;tc=&{N{q5cw6<<cmmY8CVQvGe8KZe
(J*B@08C{^##1bk(e~>eP2{F4Qu}l7wX1od{*>zOmnHI85h#9fy22gQu0D190*z4?ck{WKpa*`>PL<QUy5oBvt?z|4FO1UPJ-zRk7wYyVy7ZMNVg2E`R3o{o
xX0VW>h*?*RsQ8evkV^K`dj$t)uuNzBsJ7H2<9NR$oP(O+o9s!-W@qnkFgl_VEs^rHF$Ya2DUw|L2INb$d;UqM|1JcndkICW!DqZJL8Rv!NI2l9bND)RhBG9
gdtV={g<8(MmQ{gbEx~YDJ+*BZ64=w1Ix3+y30&~xGp+cHw}Mi$8|g@CUephYaw0oi4x>HP;1DhcHu~ERxaZMD|8O$H{K%ikYj=Gfdrrb+i$30cPQ<;rwLA*
$npk{k-#wBr_<aMf|&%FrRG`^k_K<}sc-2a>PSbcf{8zV?Q=c0bB_)zPgj1vs$zpjQq-=)J{Djry)|%gBNEMQ6^_H?d==j-Pkm2I5AA9CLoehi(DdP~z_D;&
+?e6r6SU<I2c__R(V~v1IL^@D@x=*FcK6?jrIIjOSQ)*3K@rZsSc|FaBf$MfI^>9S9`0*WR1F=-!T2NQ^2-OVAmHDLITh6ayiF8vy4>!Hr0VDAFGM>b`A1Nr
MTIAlT8@p%R9wOAF`X-WPR1alT{b5;Cm(&mOczPB1+ey{RN-k#f$2Btn-khr&<e}?E83X_hWg|ax8H~3P+wCifqcH%<vV!8m80>Y)um^e*AWU^p-Go^+aqCZ
wlVpd4@}GVs*!#dqm=4j8zDCrJe!TSmzancl;S$yrW*^|(>aCaF(HWQ)0=V#i^I#C`O7WjdS3o>+_ZSp7M8zOq(<LHL*aVZuw)K-+^lClbvDG~(-G$Hq}ZF#
dpUAWXCo6WGcy9CN9^%vO@@m4a~#sIytb|xOvlb$c4Z^wHP|`B-N{W!0?%ae(VzB4=%<r-?{p;sG^1N<r}e6ksmOOTJ?SDgZn^L~-noIBeiyl}#g!sLoBJd2
iwpkMKQTJX5sq`SN@jb@Jy9kT=t=Ry7JENDk-9fW07>!Cei>nB(DG#;FV6KtV)cshW4$CCZT{?JmYIg??9btRNmp>}#6`8uqfuC_3wJlSC+DR2cCf#?CsH2p
dCy5F;FGJ-nD}}MuHQ;Y3Aq@82cfxKCF_A0-E!P{Go}_7gabqA9|oa~Qn>v4oHxXh+E=Q!laYVY;|JaAMkvH3@Rb($VZUX5cdTX<rUD!p-wOL;Z0dwp)sAx5
FZ^l_zd;23hOZCr*CM>?Ti*3i-vFZ?4;=SIy5M%Y)Y@yUQn1ieniv&C!PM&1WXN$XoLet?;LlnD{be^+$w4C8S|uWXN7zF+%0#JfXB%EoQk8Kj+JW0TUYLJR
2rdm(j<Tjjfu7(gG$$8>U!1v8Gk<Sm>iS_((%DpSA3glqB#(gEM2!~|&vVf&B+YVzCIVlMckO%BPzn0IV&&BR$?#GzoO^jE2&V^6dnL$O;41T_idmx+<cq6V
@N(ya#k#H|V~-0s^He9&72`oGuklVbFA@@aTdHf4tZ?9;eMt0mA}qBhtok?&v2LV%Grq<a^9trpcY-yc#_4%uInodBDs>kAV>3sTi~C!r>;%vsd~)f0M+X8z
d_+&AWr0mo^ZZ@&P{f?{;Ce_+_Q5{m!@_YcaLfA}|4t<ne#V<4V~b^A`$9ZHxSoWI^X)7b1x#@2ru>65H6)Cm77WOdz5xyaeZST<bL8|;u^p6(M`_0^_bEd^
4FBhMpeQL3P7WXTF+T{xPR5j^l%Gk+=suQuo2da33{h61v)*8LU5WHxQAa!HM5KVdIW%7%3Kh|C#^j&pvYXe;U{$OBeEG38{GNZH-^rAR+Je$X*3Bp!VbNIg
e@sM`j9K=7hqI6)-TS(vKo<x0_{&(^Il?FHN9%6eES!&2rDNKy1SV6i-^htXD+^yi8Q}&v-O5t@GfKf*?K-f0Dhm`ZGZ;!cwDJ1ck>WB68~8C#R`A_U!Aqai
-fQ7L$k)BUM`_$1?qBClhbZSji{G;EQneLE3S!*%hXz47C;T^kUnDxI?(ThQ?1eXXtL2=-b&<{cU&_355SnYnGQ=Mw;I_-TwF4w?oaxD->^SQW>w`Lv({CqW
r|9qBL#C>*3g`$w_b&!5<}N)?`Mi*2{!UP%<NvNl{Mvs|jPipo-YOAZPWr?kf_VH#R7(`(HHD0Q-bUbbLUhYh)=>P)^xSk~%z##DrSf~jK(N>TS`0jH1kY{V
b&pG7$Qh#$6v&RkNsX7L;m3pUHSS7a*dZdA>Zp`?G(vIjWt4#KTOVj-?OBfCv4oJe&v$w23`{e{@5=E^g<EFi?w`oOqhl>T!VHNJ8RAX2awQXKotH@wrds&g
B_(~VsS=l-Des@KE=BcDk=7l>p_nPQqj~$=9Ygb3$Q%xXpOl2C{$K(m9fiKhxafjfaKHCONhgdXAC+Iab{<;q4qHf%S^~&#H8IYDPg7Xs9eD!EgI+b#F&V?t
>q+o+D<Zz^-73AdYz@f+PNr<a_F!<SQscO<2TIvL`nP>M(Qiy4urx&WK*l)h?qnjWr=2IZkD6d4ROGF#BLV+)9}n>*S>d3>#bOzE546P=9^ZFT53e}>FcR)t
<D~CwT8hg#44$?NN^%H8vc$;HM-g?1UMVu&N(h7bF4`K^K?0hCf<ANUTEH{0+<uHk2K%&>{ldaBQU9BU+k)VZm9Ky9T^5SK!#Mx5;j&staqwV)MmXquiPJ1(
Z{n(mVV6qtz}q<959j3Vz?GCOZnNMH_nw_rcf7qZx6pIR;2QxSn-1*w!QcW?xWR|Q?^akiKCbPk8V6pF)Nz*zJshVD$^38eHi%kAo-QidxL9*@mhP7WZe5B0
T&C@V>N7hA&8kenU+(Xtdd(MJXLw@nN@$=pKuDF+z#6$$)4M0u{m^!juOztH4=Wo1pK_1cgQK~HsnNg)e5uj&Q!HAz_*7x~t7#}G-qc!u=hDLZe&gxw8xEk+
rwYoJjX)>=3#+p>ZP*=oFn_;;HFk}1n{!VaAh|dG2JHj^3?Jk5ay<RPWKQQ1xuJ<t#V2nL?r=hz{N@Xadxq$k5YDmGb-}`3X3g~$JG@%`?K^P59>Y=NV=(l`
5dGZM76E&>u*cBdGxx>10VDH<-DP~QyBTx#b0B<6%uf7HAYo-OoM?Kv3fm0+PHZhUu&Nt#6=Bmy>aM3fbpy88>i3bAyl4&1`9r?^0?wemKO+!GmxSVTgTcjd
9=Kl`UG;HKDK3S3=}a>CfwnYcsG8jxp1H>yx`?(&{LNfYYSjFH`weZ^LO)G8VCUa=7Czk$=np6w;pTRP)ZK8oYDFVtcS`aWZCJo0WA#~)V>G&1st!6_)xrnA
$_<9wfjC}#Jj{;G8><YT_B_t=1SkUYzR{Rt?9JNub8hn3{i*Y3cj0R+i^l%5Ue5({C4<d7Ye!r=lpJcaq>3$?GlKfe`FK;4{4{BwHI@R|EBF$~d2&)E|7cSc
W>~1#D3apP73ZqWSmc1>?K%Btq1M>je?-%qsSBRjaWCySqVUaZlk2;$10HCJiUqsI;8a)kw`Dc5hf$YMe@^$sm(q;%zh8sE*)acvipvG_j5BT><-uU6k{4>@
jlj0mEya&JZ{e&r!?Nu4NPJ-(8l+-RhK!n)PJ$N^Z|m+T?bz!Cg?-uQ7%lzrP?T0N@`^7^?pR#j^a{qxtB0b&MbX$~?@Cm^ABTDK#w*#M-(l>d#C%D23>Nq8
KODLz73uF{KGLVuBXHs4uqN3j`~D42u*ihs<L_{fTb#jAxBhca(k&1TkvwP3b;-Y%&{xX1S`N?BA<fdvbTGN~<f}5?!~>SeKN811F)SP1Y(VCYK85iEJRcqK
TW9_3g#tJD8l+q~q!b2~yV2%>hZ<na#6=<JYXt{7Q_Y#)n@|(B_;F=J1Mm3KCHz`_p|@H&u<#-p9rvSL;$pmzFsLmi{?;CH5AzN@{~3WZOyB-p7V$yTj|jR5
<7~u@KXZ$q$_9bsXJ@}M2~3-WJC5xwIIPYi7(NpNyYU}>M-RI~W-TcE$)Q5nmfSmc)U^V5#yoPHp#=6;v<@aNagbrr<JbxfK*Rn=?B)eFu;FyKWoz-mrA;=Y
3v($LJVRaQtDJ`3?E*7t?_lt#xZbM_Bq5e{bmGE}bfn%F+AYayh2OCj99hb#Aa;D;rJJ7tFTIx?lP*Ch{j{iejLr^~>`^T@x}zYo|HVVaO<$-Ok8w0BxgbiK
P$m4s3~r-$IJXF?u;_>xpXn$A#n#4k*8T(}sux|RD~Z6BrjeZe7Ex$<r+L!tv@dS?u=mqENyYCYE3Y&gy<ojRt*h+!WmFwcG0P-I<NDDhd0F!DMho5>_OtDQ
wKZ4SdUy>M^{HGr#;qW}s3x&%HQ@j6_qMU4ehD!LR$qgoUTFWcLu^kq38o<dsoX_gV9(%{D+v!m-(MbX;w2IW5A17HctpaY#3Wl|jw}9)dyr0TUxKGL5@R&v
x{&2q`x<e>4^7|WWKK?zplSWV@%MR42rs(}O4gi%+L55dQ*R_N)3?h%;DiM((+;iBs0X57@QlDmOFK{syz(d5B%@~W1kcd*W>EdyE;pR?z;UOCG<Jr;=&eqy
{xs_W4udD3cCGkh@YbX6+*L-nw6Dmz;HC?@^eAfw?0t}UGUzaAR{%CnxCipxPR8!n)y2Q>=;GTMN$ZRXTgbC4tZMUjAv(|4t5@m*Vr31uvhN0iW2W}DW3&{a
Y7hDD|EB}fpO$<3x((6)v210AMHZ)eclBlE>R>A4?$o;(f9&1UqLDII4fm3cx6CqDIC?xodRT<)1>=KZv1fJRO|ccabSH*9e`6Li>^5i}O(ycB9KnTw@Y3Lf
Qs}?1Vk-<ug7)s#cAhsz*wvM{LnNdK5=s<5wu1DroygTZuIGV%i(V@M$t0`>GB3^x8^iD8b_Um0CC>P%(Cv5jf>G`c3jQH)$Q3hfa?l0f%c}I@X-g}FaI)TV
_(X)&k%7a8)app)o{VC>Pr|)&F4xssSBMGistbs9fZ(v4X!RdeESxz0Uc@RCgSU%0a$1$4Ut^^4{<H%MXlm-`<m_;Ug%JAmhWr2Wn=LO@P=O-==jT|8${u@S
_urlG!!B5$=mD=Bi=#I>W37@R?q@*4+49}TAV+*qR%D=?FNdgPy#+sO5P96LEncUqM2L&@YV}uHH0vhG#OFC;`FZme{fQ9dkM4_{SP2Blzg1~M))zm6jzslW
5mDj)-l~?z7vuCUbE-uIq+X%;)je&Gvs~gj`}6&we(uwM=p#TP@6E>;IWm`u+p^X2IH5M8;4L+?AMXEmY}=OF99D*w3jPa$`1j)Vdpe&GO#0|NU6%@kmF|(d
3D>ltSEKvx@jp|@rV1_$>#AVd?8sm2ibPf&_rq9g56D!1P}TM#;Z9Gp{1qA(WL`OS)asHW22~a!+V+KF{MP~A<WN6w)n%#7y(2(JG#!D_a>)8JXnW?56Dro5
_i}T2;mX&l4<SZbkhfzt?KO*nA176XlpZir;j-1*n}D&BS&n+s2;8x)^rz+vgR`cYPW7NBq956`J1d4@q{rD)IK>?CTi;?OBJH5J7;YGw>Vw6TH{0z>l5qQX
(Gi~5^Vs#7UEy($J5F{)F3k0spptJ-HtC@bdhW74syF`+DQ`omBlYdjlW*r=P#c1keS%|)dRnLp4^}*8pN@zmpRQIxcNoa<@aQQ-;LTgQe*azquJz<bFb<bP
`)Qun{*q9H)!x)ynhM1Q!i7nZAp)qMbR0Rq;0*UY^g{Dz5>TRTdf)MgBJPzGS}kw8!FH$2Jm<(2h;#=2s`0TxAB}E3akl~f1kP)iCM02R`0=v#US}+(Ib7Yp
5{q+8{S-XAv|z*@f9g|79%eYKmM^nhL1&1nv?O^xqUK07sU?a?QF|i)-605vFE7xQQ`({`Gr`gLZXDL_4U@tx+)!bwF`{+J3D1+iy<jGIAu`f*V(MumrVOVp
b$(4kbpame6vo2$*!XGt?g(7{5)!n-(F1%|!YyVz4v2gsE+nySif*?GqHnQ3l#M*4N=bpZA7~OC>>iA$r|k=gN(5v`9c@lIYlDOUtysQq8u%*Dk@M5k2^10;
%etaL7?qt*Yq_TjMXl;78!{JDe5H$KzU~Fp*t?@@U$g%IdoapaZwKKjr`X=S_CP2aP77*KML_%gM&N^!i8x*NN5S3P6GW9h-#<A)pe=T++LZJDU!4xI+)9y{
a>k+8Y|EJnSvdPGPxk|p4=6U=kL`;ILE^1b%Txj8h#vgzPaznF^vSJ(<t!`ox@h!G{P05DubM%_32*SOAO3t@Hx+H6sL~J*Mh*klK=KbAeAe_@Pd=xO;0uRd
W-NFh+tNbMeJKt7D{2K|n&tTDI(^8CJkP61yNS92Wtg4*IW$MS0O`$-+@|DueI8y%>q;F7ExxL~ua8{CySCX)iaZi(dMsBftlV*2XH&K<&Iw+M6N%vlE^rWj
8FHX30CcAb{#gz&Xsnra>gFc<_~oB`AIUrx<g$06{w)dd?<ezb_r-%VbfQRKo6O@LlX5%PJdu&B!0NS4&ON!TM>fyB@Z($*#n<<K_@dY8Fka}1Z$!<Rof#fD
agOWRlmnR?q-Imc_LD%*$8lDlz5oe)Fc%$-z~{HVi!9DQ`1tPc5z8`vgf3p#f5V)JVR4e-yUsMk*bAFEJoAEg?fTQkjuL2$O=jj2i6F9%s$4u>4DWtYw^VK#
?$yZs4NH&5rxtd_d2JW4ILNwxRZIXe_~caI(*#(Y&n59i#)6(p|HaDfc<d5ACmwk0BAmnAGaasGVSh{YR_O0Yh|D)L4!o8IRln5sP!j=Hl-k1uQ}5y#%hxpN
!(O;?)b5$*1lfzdY1!Ehdmt~uLOtfSIeb%Jmi6@Hqm%g4!1)RR=iFnr{u06<w70SEv2PH9Qb)#KmWRSN>6@WneL9q$uP-+obcAhZdTLXc3A!$xf19yDLS<3o
oyTKAs1JIPcPY~saqKZSR8M&$Abahv9+MY_8E^SBCYEA7lTtxg|2F=mUmH9Btr(XdbsF;9#esw8PvvQzQt)qYeXg&!gSl*&ZrOY&6m;G{9nwmL=$OBa_Ll&h
yLL!yRf0SpUH0r}+p^KeAHl#0Lr~OoNGv)yVc$US#Clc;zTW%SPEj<7HS-yEx&;jgw<Vvxk{N@u$*bRmZ+L<3aYHmUn>#Mi#LnEcsX*gWisP@7_E@wlyeJ**
3$1x6(U5Es`UQ^kcvMH>{oLDzgCq}_PHFjj*u;R5P4mhGzaQ>&^4;WgCg7b&31gjn97cD0Mt|7tg5IT!7***wG9Tyc<7jZgf0e!mf=+AW?TDny&beSDhua>~
n9jmn1A`dL85d9;vunK3Pzl9GjZoR$skkAq=S1%ZA`Dgx=SLjPk>w964hAb&J2(n32{_{Q?SIdYTPP!fn;_ZOLGJ%t_D<>7)xeX=72z_g3&bo+?<6KeJJmsy
WjY0Ib_A}IGYQy#b4p-h!5%L%n0ayr^+7aq)M+TlMWU6XkUg_6qVJw`Jtdxpej2gjs1zdJFG}Zmr}$xMcIIo<Xe3^j{L1Ct8w?8X6Kdb3onZLlOvlUxf8=mt
@B49}F=kYa-ZK_0x(mUnMFDutRn|8sFNW(4r+g0g*nl9?>CW~#5bXcme;Gz}g3x3rtI+odoKsq|K1=4Vc|1C)xkC?2+4;KW9P#A7UiLRVD+2rjimlYlad=>(
9_SKljfztOOBLkf?=hBt)v#iP{hf;*K_UsrUo2%f_go6OHEc2odbvoxcu+Xas}S;<lYzVUIYQ9I@Z!&yM4an&p}NAO3!goQ60A1e@mHknEGxkSC;R8)wRk<@
Q+_DntG^xM)VX&)*KmZ3sF9JSn<akzlV`~N6oad2lWmoU(lALqFr!oD4fRWPuR6Y7hQ~+kTHifcsL)&7Qt1dJuOrSso{<DB5P9VuQ^!F3K=6V(y&sfcs!V?3
_eRBB>vIzmZxqZLy_TVD!fP7IxN%Lf|K+#d*WD&=>yc0??_cft5(Kp;Q31j~({PAQzyIqF5+dme7i!vx=+WD9TlLIGg{O?wv>O4YVQNf4cl02={5t!=LI4cU
CcKH#w1w{T;hsP`Pgwp+akNwkK)A;Rb|<o@$QE3^AobH7hMDTrM^AYn^nFpCUr#&)Yn=UC2NLnS!B%{tJPbqeO3_rqp?Lg!_s$1iF4)$ndY)si3(?U#f#RHn
*z3FJ!;hg9OucB!SyFn8t$oVk<|pmYr5L%lW7Zcbt)<b#X0p#cm@nTLHbt9jC?&mtI$}>M?x2e^f%C;LqcYp(pi8Y=UECQD9sSkjh5<X2u%(I*P-tVv?(0Ga
DBW?p<ZRP?2?5Q5JE*fntwEq{PrY+e6TX%riw|Q|(SAHaxmGO_`EvacHCqwbd4(QRU-U2_N^n2?!vqP|DH6u60jL*Ni$7~-jh?`U<%^qkxXI<1AOFD$UgnbZ
-&#d+#B*7-xXBf-t+rp~hv?#Ty;VuRejL=|qdN{3xxrQN;JeQmwqO_9Sr&gt8~0CD8ihUbMSS|wa8<iMXhI1pJFaCSyQfg}T@9Jbe*b-1-IaxRRt6uF&Uivl
W#41NNiSrNuO-@^h=Ni*)7E<y61L98mfw3~2ame>y<(fz*zuORGPc+dEk3##8gmv<5=yuvJrs-)rM;bwN9@7jS$OR?iyQdOwp83c=|NnwpZY_mHUi$g45;by
$H!CdKGO1f5O6gh%FaaMR<6CK_OuoTY*cp@YuRGv^RA|SeXe+Ve`cdy%M&Hk`b{LVUtG9Du{dwwgxgAKbz@!X;D1HWsOjtq+Amb0H&h7l-{DL8XljjbZx^)W
2J=ug6{aP*V2jTjo}-F418}2*ZSM0vQ^;AwXF01lqP0Gk$;r(dGD$+`0x3)|es1&fw2L+d>X@gR4Ro*?`flm$4G;L)gvUCP`MD<7I{Dj*9%%9&?Y=HR9?#PS
Qq+$f&~>ujQg*}^(~=jO?$ln!xVzB~8(JH@b4cWRdnX>(j+pH(nKuSg#3?g_3NN?`HRT`VYevg}CMl-F5d6y;5#P>Q;RMsGFCAaZp!e{3^Ji5btk+Q;aGtZl
W|5DQwskaGLfE>KuM^=xWAdiH&>C6>oR$wc?a;aM)&IX>+yB+6$mG0MoS7xMmwBYa{cUlZ_v-YdpfxsI|6RMP<&GBDOH<Bu)^J*~Zb%OIM!NO>;UwY9@DAzr
oyZNwr!i;3&GXKf$e+>mKJAWD&6y)*l;rVX{#_Akq=d9IJ62~4Tl6G$u!SY1<ImAu5AEdbp_1-lb;491PsWtLRPx5+pK|F_m&00c*tH~DoM#4A=|o?rqDcJw
(K$Z<$Qv_E%p*#A*0>%m>);`wi@POxmT_*W(0=y#jY4lQ*0*+DyX9(zO;?@|Pb}<_G10<Zcjp#3lj-Bj$J{X*vRg`Y$p>C^-S=`gJ@NgW3}0(I0m0Pktb&(w
k=<f`WK}jE(~Vm)eiCF4JQmp3`Yj2*yn3Z|hY3&~NlTDjsmIr*0yo_bV@Rvz8BZ?yqKa0B-tv(hj>c-ro-l|d_ceva=dl@>6;%q+I79Z5k^4?vMK@u)kZ%9$
NiV!)#2Di2{P0VLw)KIXGxAJ$LNv)aK^=E$dd8yz2U<0rJ9e2tMZ4gre{vw)raiaptfImF&vRbJ%N6EluI{sUj)r)iVpyVrCoY_o{4OL$MC1Mc9z9|(#l5qv
T;XY%c$MdJw6=qY#801I-@F=&kC%_`^|Nw>TUEvj<ln-JL%N!@D{8PFR$e*q(*@$($5Tc`?LjGN+WKMH1d&m<Z#>k|#)|$a0qSW2MiS4*$D8QERKwuA@^l`W
+Qt4E?ufuy9vfa(GQZwF+c%y6ED+8Y9ENGGCZQogNX$#l6a2aqPh}(>apl_Jm6(@TvF7-mlD_2%^i>)Z)ks9N*A@<T_Y%P#Nr+kKHAnfWMgutxJFv9wyX__H
gVdtk!Q*D(7-;`D-BaR#G}E8Yj90p#tNw4{YA6v8Hh4;2v75l+9j)6ZvS0Jvc>SlsECvBQ-!ANT^2B|`_2+&Ml(8s~Gcne0jdO;M*VJzXqn4#ow~)aN`)At(
7b7H4@ZR^3v{(Qx$^N*#u|EbhsmJUcnn~EGyf+(I?*&uppMr`awpfr-BYxg+fKEJL*j*V<lyNfV)Jr=fSLGnKd~YLN<fMb8Zvc{F%`KPmO~FIEbjCI1F8Zll
Hhi}u;nOnh%6za00z%R0uginU-rSj%>_UPvJ6C(+0s-?e50kE)BVu(?EAz|&7t|)Xlq=A<;&ZGWeaMgpt`=3CO|~b`w*gI}fExMp$Uchj@Q>&TdHjC+sR@Ej
-*Fu7w?mt-+UM}c8JMSADy+yXh5KRBGp2MNbjAF6%lxelQ|5DY*>!esen&l%Mj*hj*oa-~vpXVHIp!}b^y2*{|AMXw*|Qd;pOY%=ke)`TZ+^NR%l|31-*{|`
UQWB~yNq1nE52baLe8ZuXVSM<Z1rd=P#!pb&=&EokLJw+y+A2P{XJ8Ni1lMrHKnh$p|i9QtKp>!qM&JGa9cFIw+atT-BZAx@KzQd>s)MgG#$9ap##m&8nlP#
s}bovFm@?U7n3D1yrn(57<j=^lQ67^n!`Vr1iGEExi_leT9i8&wFJM?Ryrf&$Lq$qg+%f^bUC`1T*CRkRBRh&88~pVU&kuzE?k-p8daX;0+p58-L>XOxSPy*
W#p52f870tmAWbvST4VmagW1n$d*QIj~v!Kd+x61lTg-|1BpT}+>6%{Uf*W~ZM`Gq{-SQ!J00>qEbR^iyyAyX4Cq69ZD<dltS<sz`ezz{(tv_ItzyHvBicW<
YRKzrpnBg?4vRZJnAzI#c$ecXoT(-Vo;c}%&5OJDpSfX$a23vU`wl0h=^m@n{UDELl+B_Gistz5HB(CEmJfuh>bk5q{QsBV3)Iq<Hy3<R*cVMXziE%bJDhSy
#VnC6b*!Cg*bslB{`>nU&=awJS7lD8dSSbT>G48)5;819d*$-Wz;g6$!tAX;Y`)6v7Oqx>+%Wq?nui{c(jPduMtqFaQb)ZH|3VNglGIno>ItbbktrpzM>*Bp
a+sO%2a9PHV~bER9&(>@m2vZj5X*ntYTGGbOlnUV8X}-aXVPCu)dcUq30PcM2uBx<uUiu#70=&VYU!ohV|{pM&4!CDG_4<14|n;()>-71%^`n$tJM~|b3p-7
Vuxt%{!_zP-#PtPBl@^;vtzSv%MA=70}|(!T=A#pMEPQ~Bkr=vs4na_z^YyF!`ydH;Jr8;oucG{zQE<n>|afh@56D;ky{7+)3-j^JXXfvxc5?mzXPxc1Adw%
vhU{BzOmM?MH#;#?L<x`yoj82@qY-|*?E25e3Z;-FQ1ez?RUU*&ga)Z9*3b?jfFH-C;|B|`qz2_9bxB~_N*w^0<9k}60<o&Azt!&p6#_aGMM93t9|@IeTk>c
Wl0dXTxew%>Vlv|<NKVk&K_nic?rYSR(MwRXY8Sk99*6paH#Z3fswcg>-we-XiG1WWaE55RS`BZRqKH5Cd#lwgm6UWD1~u%5|DgwRo+{|6{|lUsCgt1kseT~
&=PBd>{oxTb`1qVWTac^)#xD9|1|hcG<oB^@d95glM${RzL>^8rjCWNyg;*FJKRuWd(J^i9*4w}=Gl)^A;=aqb~?lYJ*oSy`{Y_9ft%{{*jsxfoV<8|<GBZ3
&}t7i>)S$t#Qc0mQyNYbv9e5(^VZ-I=bPtUj`${XN!hkY2J|xzXU~xN^zUUK|K1;^=rcY)b@+rePKf_}Viu-__U_BRJMLQIf&0M8Oiu^sIrCSart}0wW!Bv8
Au^AN+mA+u+rpvn%GI&xNbLI>K}jp`imo&2zog0D<92Z9ueMq&rrmq<Uvt^x$(2#TSs4Pl8F;pDxf38fa4kBCBLEEpQ-yL@bg(qfA1)YVgx`Z4dzQSdQ7G5{
da{A+({Ve4EhlYYQtYgwET)Z-H4B!9dITsE(#$zcoUteMeeQ8>Ul4u@mezDQLxXqbLU~C9qJC~g$tRQRVlC#>$xr4`R5<i&{IVT(PDZ8;F8<$h?m`K<KLj3`
AyxWmRja%whBO(N-YQ$6X@+uvTdWLQ@joVX4fJqg^2pnkZ%*KL)a#Bc*FoihD};6tKh#E#C@2*X5xLxr)>1DVp%QHzWOj!&`$*5z{!6e}D~gf0U;!0Ml4L@A
D6Vyfd(zZ9!OlK$x5x={KMiOxJ4EIT_n6BS0_wWBu}5L7d_y03jq0I(HTH<P;F@prAOlt3l8<kz<m2e}yZkgWH`GxZwbIM`A@Yf{a=nck{^eT}ALkpu=l#8<
!BG!fdia2%*gq8UY648UKZ@Z-9iQH9nt`sK0;l`6!Pw+Amz<FZg1NoWMDq(b#2+U5GPMLC*=L`+WU4<tbM8r6X7#|kpeMY}`zBDRwtaAI+Xmqm+LY|XE#Vp&
);|4#gsBxN-95j&ahUZ#yYc7AaJ*7oof;F3#^AS#)cgE#jGOuRW|SX{U!Il!Cgcmn>arygdn8_bms;Vb@WFXE;!Q1TKRi)etx4>##coSgzqx6DoI8`=ITVu&
hySj$1zJX;{_5hN7tBP=d29Z(YtKPb!t(PNLJw|S`Kb1`m+WnfV(o7}gdwYY>7YP}4jT4xEB$uLhJ>tC*6o0Jn5WxR-gQ*~t+TL*nQ;&-8HnSnBu|K}tsU<6
^gyfnN>%<2UzptqdNpz?0(n`@{GT4UfYxp4=)6z@W+m1-N8)`U`0sV9m!>z8k9h95rxc5t5{C=S)}f$%#b^*7mw@Y@vyVmELa<)@e3N0o1k)#el^X|!V9f07
rKgcm(7Ze=xK<tuQN6qAhi3yY_T;{(_FaEWf9BsuO)9~bLgN+Mzy6@)vO8Ka<A(?LvI?*ERbbC&%^wTq{wT|o7-l`<iXW48OR^STh&p*sy{S0?+BFsnEWD8z
`=T82nVf^N)Jk^_^k*T+Sz}m}+6#YN?yl}&cf*DEf1|irV)5B4n%1#93UeW5^!!sYpqU_vXqn~Xp)TWhLYE)>)N1q(T@S$7W(x|TOOcp4aG%fSgBxsH$JlC^
gAg<=d6NEA0>tQ=7^(luftfv*f1Z~!JlUm7XcMz=&Y7Wt^#%diM>?Kx><owP`p>&BABBL_kZ?MO{wZvfN3SPMCF59uX`*^lG!}1IX*C9FA;)x)z}gyuGl!UT
PDp1VXCH63eI)^RFKz@(DP&^egXX+XO$3N54c9gQg=0K)Rpy|&4|={D)iMg{g6<wsB+rxxfzL;8B;E;tVB_iWqDPKcIDfh2PuvAm4E2r}NQz-L`^$NyR!ek_
NSW;%^ai85d-4@}AKdsNSlm1ri9KZ^_oM~LJ{S8ydW$g!Yz^Xf{k)>^(vU^)hjszlu8%Rygt;Mftjhi=w*xwtld<O>3C*k{3`Pz%`0?O|WVUuZys`u9)R@D;
edfonDjIcooZTjDI1*vD_;-MpDGJf{eVRQvj_^E@o<hle6~BDl;!ah`Lr&fBvFv?Y$SZtXD6e$^6LspXs%vVvpG)<UsZ|g6Z6ID&n}L_-NCoFN@*xsyxnNhQ
4O8Je+*JGhpz5(Q=AEL7QuPzY-b3Cvc;eT$?0{svKOY{$OyWl4jgGvk-~=ez-5B7^iA4L%T}K~RJv@4Q?ekfk4Di&b%>AvAg-!GJ$<W<)aGO;9sB6^^L-p7Y
)nYHytZqDcWK6`U+-c{JwV|-zb3wOAOB*j;?<?gOkl@#nK*`h)1O>D2jE5gO!8J!VS?-A{(!+%RD=SGwQuBpR-~2PtR^36Aa&pBY_aN`weIxAqtX@T0_Cwo1
p~&1HQ!uv|+Wz5o!D6v<7XzO=WP|rLzq;{%_h(6n`%jdbx+CM+-{xJc)|mOdI7`7<h&85HGW>i_SPQ<uEx0oo-->!TcJ}AuS+HMK(7PZ|{@ycU$nS~S*f8TA
r8a0*JJ}K}<qdx$t!f@FfABh{e~3Ndk2=1lry>I+9Fsj7lMoySw(wTLR52H%we;PA3jt=VHx`Reg`;6j#L)AwGfIqw6e$me;>F8?nwsk=s86f);Ei;}@pAJ7
aS3N!Kw0hmXfJ5Z3n}YF7{P{Alt7Ua4Wq`kO7*2sM0x6KHIsd5F?p8%{hA8IQWA}>Hy9vRF2*;+RT>-j=r}*gCE(WUzvYaD9+2E$%0f?ygrcWQ=e0e7aMvmA
VP|qcb#;5_%Ht@w(=&!G2RI{P?2sE%yesyPe&a57R)W>*H;MbG-66K;K3&3hS5WlYOp87%gUr>Kjspu;pm;y_lA75H|C!jEXwj1}F&x6-ZdZ-2gYkPT-ew|j
-e7mtZ+k3BN#)rU2f(;(+<L$$2!iYPEL!%JppN~R#soVF1Kw=!mcGS;`|l>7m@YpooO@i$l^mflx%!}<#|V@q?!PO_^g*dJCd{BI2=U(Ej*{bMm|UujQgx4n
gY|NRWr#M8S@T%&-it)j8NWw06iyiYC$QKX5CV#~S3m9Qbj7aTZ&AkdS7E)PRq?et5bejtJB0uEg0vm*bcoX%C6wxf?<brQ+d;SY$Ln-lep;Ftu3>=SLpi&t
4?Dr{&uQ@o6gjw4`mAZ*$`nhrYjy5XVaN-r-nMQ^g4Wwk>ZMH&XiZ4G-)j(xtCdY&tfFc-ol)(}$C-k@wdY@zMspy#H=pwYX8`&*&eK!q$wJYxOe|i^6?Ex|
F5L&L;1kxILQUfWX}|T35H1JMQaw-pe%BUf2(J>E3mlM86-o+cA)>0~2t)YiWE7fp=#T7<0xvV)7>{Kr1~g{m$~Y6D95NlMv>Ai6Ph*J$VPjNxQ#xo4hQp)%
Z|K=D5<ctidc$`%9;Y9rZ{)mofQJerN9BMOSRc4njwb7ZaZllz)tWtWGuz5*<%l@EvPk_pPaE&PH>!+^kzm7BCKOdfLf<L-^?Z(0RQ(n3O+M}oyVI02{J-Mi
$avvX-nbte+f#qddHNwZ(qOMmTNDa8c8quTlR5m8;l;CF|99OmtJD8ke6K(DndINGs7r>;TJwZKtQQ*EK0P9FCWF3sS9htT893?_+8-E@@F~Rl?ktlxY-dS{
>2blR{aEsXH?j<7VHDY&kAo4{^o-jjD*!daG0%S&>m#`0#jykPTHucKrBflffJeH!{>4oq*aN!V4n4b$FnjCEf$LtVZq=o=cz6Q}&;NAYTo=X@2}3!NI$wBi
)EsX}4aFh5oqM5G2?J|uzoTUl5Pno8oXqNglD9nOLJVYoci=cQy5)`o>tc$_=1v$p*0&jHPK56p(HVu_Alzl=b?C|U#ppQOu@iD(WRH@T(6%JOrJH;CUyL63
HI*%$X<hKeTlq>lzbn>kObWE;JaBKI@0u~WPttl%w<`3zBIgo~ecL~8>@;*0)?=u~rK2UY4QoWuBt!;YqsxMU$?W(jPc*cu8wY(vJn^99PI($RH*ODJuiQWE
iR~Lj9DntLVPapRU8dlLp6q4SM)Eqn6Pv>uc#DA3mYi)*Z6cvkEbaE_VJ7-5H&)J6Mk4lyX2I&1FEX5_cVAmggjC?xDbwE>WUnh-KhG754dqvYt2?6c=j(S~
E+q*NUIZ}JttP{SVPm;6%^Uh|4_0gsi-2%3I%m?7>|a~r<EEyb_%dU5INF?mT1UPlNj+~E(5bJ^cG`o3j+cU8+#ED4=M%)v6l2UX`E?wh6^fq)tfd|a0mp^-
9~mV+I8dYR>hsG74jw73f4D<Xc-d*Q`MVX?f7R5JIt_4du2=p+oE_AvymCgW{1KM$$m78?cN{V#T#Tde0#EGKP_Dlo;M)CEL@@mdq#Ir^IXC3uXxJ_DGpx0U
G!DEMVH%D2Q$tjNukvxAVdBN@YsTomW`0D3(hrj!l*@A7Nw_D+TO`xzjSlsMhrLJq@Ty9fF>W{-b*eIX$GA<wXZyAD$5khsdiJXH-xLuprxxS5j54uPTVm;@
r4~NEl~?ZR2uHn`P4HuPXPo(&^0Jd$2gL9b!e_){aCPWQRXsVUnb+?ZzWwV9hA%4DUi*hb#-!1fc9aBnW>+84VSBhHdH+))bJ?^|JY}GoJI?Kwm#qKc29=`;
ywRuPU>;ou*K&KDs5>i2pBfF0<Z}bYEq37hvYGNhLk-5wCOb>sI^(IZYqK3|CL#pyc<gL<1ala#bh}wO{KDVf{Y)K&&>cYrQpO}~-kVtevd0iDHec?%y5)jr
_g!Xm3rt~|{gH3R@FMgRxy7zkxT5Z?;-v|FEl{e6hIO!A2LE;O_D-HqjQ#k#VxJL;o4j4)d9vZ4J`=E5W@`niDS=|5lQu%AJNKUD%ElgPGf}qlL~zbEOTUO|
gurP1#ARD&OrOp<ygn9$3(w9mzK{=q8!eYufnX{^F5Yb`Y_f%7@6yZa#Xt;C$t+uqO5s+b*p-lnvEXUf+tL3t1iqtGLW+6f7}N5m{2OJDY}fqj*Y_DhwIekD
g_t8kJZ7}F#v>8*VS6%cuNv0A9gm(`aRyV}iHq8gqv6ZgMKrt|h=Xj??+b3lLaL%d?$%FZOeDs4?Gm#>81?hHmnZdMf1mxymvyp#9$=ed_;Cq5)$b{T$lTBH
wD@EyO$I3EuL*hZMZ<7A;eZfhJX-u#4d)HCp~kj)@aLc(lJ?x5WPRg|2L5r%AZA^(FTZ)Se#ryx?H8W?6()Q1K6=3hP9sP;7i92}zaN;pS*uo;il|JLN;88%
Wb@bjG$$CMtAM$Fp*#m7e_w?uSqH%R`atTfWH0QxNcH}ZXEd6gM!t3_bq7N&>%h-^b9|@OiPFCxfd8_F)edkJlk1fI<ek?-|I6<dn@!m}x}F$kYNCwvCt+!{
WrtQxIy4HZ#iONsKw-ee^m5Y`aUQX1Q}YFI&wAA`_`(z6^>r^BKN;h2e*n|Uh!1jOu7`7*#^V10P)h>@6aWAK2mk;8Apj|(_QpRT004l80{{vD002i{ZDnC@
E^csn?VSf$6xZ6vGr0n`Ybb(9RRk54fC?&>9KZr%K~WG9X^LP+>?q0_QA7|GD=GoSSis&RBE~PG+$0)}rr3?WT{T9H<@)WNL%CtI?ku}I%<TSu&w2D=cIM0}
?>RGb&Ra722K4DSnE7bRM>CoYkD53vu35)M%{q_h(5zLXW+TSOO^S;c7dAd_c+`K7_lOuXF-m%TVoXGAl=ScBZCzToYUJL|u~nmlMpdKsM?h4pW?4e`y5EfY
+r6*HPa+M*95WkpYzjdSD90s}qAtv(VVv9TYC-0-K-WmLGf;{?kMmrbbsVlUa${3G!K!#dKE??EAE5S68)CcE-h^?YCA{78^lYEVSoEisQS^}6WAN>`@_voA
J(f`8WrR#WPygpFw`D&qM*jl<stx0v$<h>ud5G`t9+p{XcOYnDoO6!MeX*WQyv3aAI4D%1ycPvBqxJHmbY8A?Z5>DJ=WBMIYAsQ194&=B0H6h!vHNHv<c?>$
&$dFa<2^p&ADMG&2XQW6@uR8#><~P%{Cv{>X}tgKAf@Z~s>K&!-wz$2@H)3`zkOHBVU*6zmxBy5S|*<gz;gls9(kC$CaJ00Y>Ybql)zojdDFF9%Xk|%Mkn2*
X#&Qny!J7dNYeyjdyTQV#$@r{PB!d)6|WOn<a$bzFS+M9_3iNH;pk5wYN&qYT*$GK^k^`2(huFEwWi&!6z+~+**RI~<luIzHM38+juc(J>v!5^(rH%)#;r(s
6eMvuyC@0$2Q<JL$Q1wpsE8HpJVZ(1uu<XvUKOu%{YJk_S88p&v->$Qwa<H1xqc=cB+?|ikK5>##P*dWj8OYW+59K5dSstf`M&h}z8dS7`fR<1DV={`%e#9e
eP@ur{ykc4<y2H}BbErI@2BWG`aA#Y?dQy0(wK8WRle^#uh18h_qLRKeb8sM>m1!=CiA=BGA8?3H%FP{`~AFEn%^`apG8Hyc`#sv+S?<i(4m&(n>I4}ZfMf*
h|Ohd?)Tfz{H!0|N$&YHzb>5f>-p*UUe~91leQ^Li2H^P%GaAMkF+1y|9h2RpXkHb=_R|8vi9z=-??4et^R}_@5cQ;_RQiM9&MC7U-7*6oj!B-%$P<p<(2n4
tw+Q1zej{J*GAuVAYA*0_KOqw$R&T&kbvfL%eSu}dDBn+ZSfw(6DN1SHfe9y5bX^B5G)_tnqHZ@Jb=`5e3bBN=_e%NWZmcH!#k1#zj}<hKEZ|P1<iO<nCn*h
cOx=qK=Y9^mbD`HeMYbJXm3Xn0_(jvxw_&#gZ|xzl>Ogno>gM{L|2krY|&)UrOsrr!|?^5g}KYji!?uD@~<lsLOX76Kq?>K;=Z@<#lt2t*E!Fzm{c&&r~LRv
3y-Z#ZVO`iXNMB&`KIN^>Ge&@|DLAT?7r_KpVD$xyw^B|_}>|4&{gI-nqPPM&)4Ivb~)a6DgV7;+PVRMdU}(CZ-Z*sIQAs7O55YF`F17edoAj9@~IsuyS{|2
e>2iO;fsH)e%XzbomY0<@tEn?AB^cr!XGvI?V5k5@^X29`M~@4x;`@HOMRJXa@w!9%<;9WnzcT!=PlEJ8`7-C*Sus|{<nGb`4f3xdlH(DFZqO(6H<O1EnmfZ
MOl5z+O_O|T28mp{`D@EgP!H}C@Jmt%Izc%nx7sCX<yv=)?HuXH#q6vpa0m61hap)s3ebF$n%2x$x%N3GW~67cPW*r{jtWL)6*J{iGH4HNj@HAVQ}|}o!tA7
ZCN%Kn>fk+-8La_dVU=Th2=c{@LEl~ZQgkA0Docgu5*d2eA@9Y7dN6N@5xIQp1%r@*~~k==Tk*DPUw307>sWK01ydF?R!eoZp_;Nh#2S%3XAlN*c1JNyP-dL
VkYzH(mjQm`);E7t6#4_v-jo5^xp?UP5|JM66@?sOi2R99RNTsW?@fF!=7%$JfWjd{m=2D?SHEOIkC#+Z@|j`a>zg6g-1>X8v~u(bw=J&=Dqsu!Y#aq3U>Zv
?Wu=}wtvj$MMzVO(&eZAAK>p*S;Xr9!9>n`f@wcSn#Ak>*dm*ercsdNDpJ`F2>kE3m@LySeECUjpuk@%(C%@vJ8p7fe#lVtKL9{5Sk?T_vgUVrn3sU4q45mi
$5Yp#>;JQLcKu(o-!J^fZO>%m)Gy%uzF!#Mbo9)cgZV^9pzDB$Zw!DKpwERZ7_Z<j(EYQ`kc(<2z_H0M;YTK-9{>P?XUX+Yy*;@zH9lV%v+;3uVO|6PXbobG
V?yKQOw+pgnCiR04gzqamhH?Z#|(BrUVu{Ad%LF9BN3vP8){!VR_S_BKl5>FufMYU*{a4rqxx_QG5*yAjY})sc#PDK5cW643XO9+PNx4NpD*h)tIb?uRm4bB
Z;kiwHxW$lfmVTPyH9-&Mw-O$U&GvEzcdALj6*En-l6jE`ybp3nrbfJzDZs7AM`#01k(@ynREEIgSYVPT8bVrdknsvg?SVRiivEzo>0goPf&llGey_l!AJeG
FYJ)D8x3KPFN6F5AONyhebh+K<56DRG_M)89y#otPq<wlf$<Lj&^hS)PcG)aYR05A9_p<Iy=LvYIUoH2001Ikc;h;QPkOE3ecrZ~tly@8T7u_ADR=T_<SsuY
K7D+|J5dh0?>tGc?PXZQQ<@UdpGxwE)dx}BZd4CsFxhzH;qmC717yyp`YQek^U+m*3h<){#7u)2iCK(fiq^}od3;WT_?(rz*VANYvhmL3ILr?K@W?8+Qx6X4
DgK@_Zp28FKY5tCCaI~LpUnBxzpSsqe0FVdy?H5mkizrYrZ6Gy8-lccKA`OutTXc-)i(>kJTJ2E(D^4`p9!+fM864~#MoS>rnfKWeu^t_KKI8y?*4j<#;jnz
9Whwu{3ix)nb}2Q3iAo-cuKM{1M$&?#;Nn~HXQ0dfv7jX{^)0s0KZSViAnd21DRL3k5hT;pIrU$ciek&O5qprqLDMFu2v6jw3Gil^@DvM00L#Ff^mucvfJT5
T|HdC7w2}M{tVJ(&L6_+^&x&Y_%Mst-X8tOL*3c`3_E+N;{spR;}ezb@4wVvwxH$Z(K7itKI-A&dzkz8TUkCPYW9mx{e(^<^6aQNuQLQQqxJHYw@Xzu&RLmW
EJiB7RF@xoMZB3cs$fz^GUP!?sGooA=jtlj;{WaA=#Ygq+z|I0008)c`hDPQpHRL%qFCdeQ-9bwcy4|0dpeewvELnmM3JA7vh%s-Z^X9$1dX@9nVRDtm$e5Z
1XP6pmL$jz0DyOXH4M0KzkQ_6%s;KiNBewxaXN4HJ!kHc#+*~%x{m#O^OiQthse#ZCcEEQ5qH+jx~NyY>rj>NPxBpLo=?%4<c{7;<iAJJddKMOb8XkE=8^fs
)aSQ!(q8i)n&0msH^2FtndVW+>eGv_>V8zx5z`SJT2%O(uBtz98b4XqZi4y~4j~`Du0l?25;s{VZt6q0-EX{td@J;IX@3N2ynhGHKf5-|8UD|N2K5K99{mCU
003wZ?*>CB{m}gYp0_+vReds0HLThFl(p5~-YaHUL>$@#J2bIE`w!oXVk5>m=g8a_ORK%#=zi2b$dACF`|YXd7r+a2ed}w;n*!8r55gDf-j`B0ZCN%z{s0gV
aqNr8E{sdHyw8Q89RZ8;S9jMh{1pDev=ti1Yzg{T<S5d^0e_FBocPkzUSt5;vzk%W|A4Mh=C^_C`P3e!`u%v&a<5WbIpF_6+z5Rg=u=07Zu4evw<qtx>IqDR
JOWW+!M-19&d($E^?)*D-I*>;iv%YxRE{TeE|(2UDDU;4{(&p(9;#{=PoHPe7`Fgu0N1@U)@*6qm-F~)m)PTs>ko?U2fIy--ulaT9tz^$MJmkaOZIxjdYJQS
tuZS6kM^ufi9P<^mvZ`D9+ymtx-gfzlD|KW^)a9s;Ex?>A3Gt>007aG#rg&IC%Wbb9sr<z9$Q{0jCiyL^9bOTI)hJoneARISMKy47yfo5YP8_w(ps{9oBnC;
_LQ_975=$qGM_HpQ>gy+`^?^#Bh!B$t~2Afg>kkW)52eC>idc1y`4+y>E1kir6}cKU|Z$=?`|@TsV(_37SAmJ1W08+#&o|s1LG9{0JMb7+jHAq4(+Y_ygogD
QOK>%3iF}y2|MCB0)W;~gz*eSfemY?pAUJ}5#}0t)oeUBoRmx{zVXWd<=0pB%!0xtdh+A)RyAK?wiDLlRAsEinJ<~KqqL?S8>l(Hn!2F&7z+sX!%(rGL-ij=
Lymcsk13`7FhO+fZcr%>PF3L=uXEbe|1ux)3A7rI*~~k==hbJo-F!z*!MtZFq8n$l+*>g}hzw))`6ofY0a5XvIfrjMcn7E~w?grU0{{R30PW>u?lej76xG*J
c6dxZ+;5?4c|P-P&pV88>L!!#lGolR3cl)2vimtPwa>Sa`F-ET?0myv&I-$OcKX7`>kb4F8mHTx{PlXGvG=E1EvzoLxud+^MbqN7u4J|9ekf=hBTv!3A3k|?
oJl9D`}eNg?|fkvBatTM;~UfWdJmcF(!QH|JpKM0nd3ChNF?X`(DN(o0BAX@E^pL-WmkNMXgSpGph8aSeq;aUMMylE-_PrkQ+mtt<FC!vrXTy$hqF9*rMwAl
x=^tm8@*2_^4Ib-C92h@UHSD7BL|=HPxX_VUs=80RH|RDg~wJVS9YH7%GRqlq4iULoC0dk;X#x;zn{YMX+49~Uf#-bLG6*e)OH`5UmMIrItJYboQZx18indV
^c0Mo(fxnKnFRm<pxbn3{XDPa{dstH_vHiq?ZvOh+ct#>ao-SB?e$0-IP~*`Ly>}MPfL#l(xh0w)%r(IX&PH?<(8*cK#xc1%zNVFQRil-TP@YO_hQ7Kssw8v
t9$W5>3Y28{asT%?%BMxud?x8sXvY77!O*G73>Rv7EptoPru+@j`u&f7c|v;qF}}k;x&92`}gK8ZI;8{PUsW`Go$tLC!?SK@|<3a_5eBw-9Ov~`BxIL+BsJh
8n<f_<OHaN<-uE|DTyq(9;&w|m%sHex*i!SdO0@PnQXi>Ia2WTkk`NT_!!J5l`uA!7@138Kddc4&H+%JZwG{Y^3~yw$bpL%JKB$JMl7BE?p{fd^+z?Xu4nZ2
B$M*vzP;;|TsG*$dHj{w4?KQv38VJ$dWy}r!tROQ$B3+JIlx-~nXSzIXt`UG;yNSTH_o!-TppUAGw1VrT=DMT`kWV`<!MabIyt%yf8j<pF1?<a<kLrCK4s@u
KAv#HrrE!yxsbAY&~lOT{L7AixF0+T?a#JK_eYtWd^rEcL|#4U@2$!1gxU>OHgqT3Kk?4Gf3w1_f#&ZmlRs@YCz<1g#i=GWdOIkrPr%k=(X$v*+WuY2&-<n8
jpc!f73Dqs_2!^^v2Nt~CwnHfoaIKo>3VdM?ZO~(yl%+Qg4RZ4ThAZ<Xz{TNnNb*!T=SxQdpmoR+Nqf(JxSlKhxT`$(~hJBEWG^aA6<y;tfRNWEZfOkH@(Zv
v|(R65$j7ob~asKU*>mO9tWA@joADu>~yI9hnr&g^6dP7hQ2Oj$RdM#zYl6lXn7rF%14jaA_)!tsM%(dA9wqcq0U#^ukS<-t{(8<Sd3@+eae2Pe-9uQC!XEA
y2Or@onQ9*e73#%kg|3xJO0!CUmC{_>q%&Pbs<|m={uycMfcKpf3o&5wBH)bIbK$-CUWbQJ$3xeLG9X?_J4n}snnl$a%q3k@>SRyw(unvZGZo)Jl|&%dj9sr
$%kC8A7lTMofT<W`a5msK=S81t2+C?_pa35-0c?~O55l0(Z^5RH+L?*PhFYo7vwuO&sg3E?@K_daD3Fm!}qX+(EXPv$hVeJvzqqFHARnV(p%P6XuOM2XiuJ)
c#ApJanK^U?QN!_KVzCd^jgcfQn)*QW#?5@ZV!uGPicxTzy6-%)VIT%SDzm_qiSA<HSNbplkoL4`0l5O#OB|uk5ax}q7OXDYrW)C&ezSoVA(KbYHQBtN7ox{
AUQ4H&Ogib@%^9wa3!=HpXvO(QPy6Wr~fGaS9mwM?a^e^v#_sPwj{zRubtB6uo~Co*SejWlBRcmu`|lG=B*yoKUJK@?sv5yb6TKloW|>O=U>c<v4z^Rf1+!8
BS6?tyPT@)S<|==D~NWCOYFOsG4A62u^0DmLwf)qKpL?A5l3Kr%AwIKiR~*%I_LJGam_VvZ!~r?^L0t!AoLp$1=KI=ZpbUpDyUzSp}fmA-T##zXT?1KOvC&$
t5q)7j}sCKTNH4w-_fYhxy$cVtivVxFm`&$17)rsSc`Ed%jK{4h7-xV8ZT6Mj=pE!8a(mST)*sRe%24~rF8!G-mYo&NQCCfrJ&xiw#O1`ysWst&Bm3xXWs3n
_VPHtpL{F6aj@F*9X8v%mDTED<nND9#B)Us)Glue+6Pn%-|PSE>(gG=p1)*dd&iD`eMr%m70kCIRQEf3>ioM6hx)@`>>7RlR!1jia&8BYyvc^gM=Flv@LVs!
Gwh7~_?wk~r~MJ3^Zh~Nzf^xb_qE5jjo4Tt6VCwvJ^)wy^oOZylA5}W$9U3GxW-=xLhLuh%K+L1wf`2MTxS{Ibo9*P-(T@v(=!K$+<|{G<yfzDd(nNJrD$gW
c!1jd#$voFV3=i%O>H99RC_ypU*ea%=h+7I7XTvS3--NLb9M<f)f`WSZ)knjiN3yceQ>zo>p}JD=V`2*bp10->HM9#T0OYYE|^qr|5dE--m~iW`Tl$Se^<ZV
^7L$<NY&rtO02UlF(pIsejyO3{fFlEJ!xDT@$NfN`)t+6e{S8*mosgJXD@Wl+$D`U=c97DVw`i1%zfdHaSi@w+{-?1H9a3t{ZIA3r&9gTSaO@S%a!%(>oJ>o
r}rH84FLQv)Xp|T<Ku!p$EIQYDWI?A-Mx~&{KYG}5jA;FUaDe!Z9|rD^dA5~25;6c-vP`I0Pw~cj5i%c-FBkZJntXT`=!a;uXvrq!`la-kDw~HmG;Nzciu3x
D*!;_3~yX#@JTOOKS1%!r%U$~Zo)jo6L<1v<Ssu2d-lo<IqIK%VTb19hSn6w{%TE;@AE1P;@c_5Ttly#jpt?)_2#F2>&0U&&V0#?jVD}>Ya4oILE#cT`1ce2
cyDvM#i-P|=*Lp7ru$vAiVQ}3XbBw_{U%Mxg6mI3@m(~n?+5-0jrSO!bbFf}wfEW@m8S82%F?jiKp8ZyR<!8lj_R4jVEk}ogeg-nDiD6IgaVD<6-qw*yZrcG
eIDnzH0#LSo}|H6!?=;d=8)o=N3SJL4}=_O36^Toi{~5aA3cn>a(}k{+p#hG7DMiNVJF%d2$B^uEFunViXcy39cR)>XY#9Iz<vAes{0XjUflEP$Ey{uv(#!~
b-B$Q<>%|B#cN&3YRx^j_g&1+HykE^K9)9`(TW)r>z?7C@A&fKI`#X1ru{KkZoYe8`+Md&bNA09uYVQY2LM%26mOa8MI}J~004j|<moHPgX-02k3J>c|4PAg
M_?c>?}y3ydBHkqQn`Ju-MjF1r_V^iwX<!m!<RG0a@XIpmaN~Ve_B>v=zKMRsCIqDd}q`?c+sDQk&sIO0001J4OAayCEjmD3XPkcf&K$JjWN4^unXg$nh~1+
E7N|AG*vyG30;@Ufm{JWAe6Y6szUWoD*RVcy%dSudU?E=<rMXg&oK@~&bm9(v!n_ASk1`uiwlGe^>0)Ceh#RgO!zMX>Va$gV4xc28jf$GF1%9S1UFqM>)*U?
ouv1jkMtF-oU{t@zcbEYfXd~6W3rPlZ`Aibe5EMmU|?JM`qi(^*QOu)v%=mm$EaUmmz*Uk-=F5ozx{Fx<7_*o1%GiO81`S~{f^MM%pqzkC)fDgM&=S}Qf!~C
?(*?PS^a3dkT|V*?$G*F=&J;?{+|XCRr3KrmuSnf2l5ZJ3OcU~FYY*9kD86~1_0m<`u#FfYv!E*mW@O`?E=4|+5*+{6R$sNs^=y?|IWrclcgyR<46HEEQ`=i
rI@qu@84}+&wu`~EJ*$HK7B3)@LnIRV~OiIudcYfnX6eeUazYKnbQJY`P)ah-D=J36Ry+McE1wq>`P2ZAEIh~{MdM&Llj>3Fm+8*Q#axH=T}s3x<6f^cjs+#
Eq{6s#dh(PudB>Ko1_Q*56<G;9&fiiJ=-Tzb$|1f`yJ>ebeh}(InZ&c`h7*O)17#0=J7cV;&Tw!2>?8gq`YU|8a(lnDR0_~&5p%qx$AiyjS8K+{LVm?>)kKR
GePf<$(Wzy5NmrZp~g$$`hj<TKlxUCWA*v7ruq45ojeq8zCM0HY9{0y000Dzs&NuOXRlX1egW5VluRkU@yh_kcD6J><^6!KOzc(Ua83RDf;4WS=KQ47_>ZFm
``j(7KP_i4(KioIWb}Q=htC&(oJX5$j5LiCO+8-fFI&*^^1|x3W31sRO^KrK|Ds@Kv|c{G`@n};YTr|u<_KO+%)*|UhCN+gUQV+Zi8M`td;^_;#>Ys5{A)Sx
dd{1!-Kx6XyQb%J`hKLmp9i7%srdW%q$r&mN0uH7q-luG&mUCJRQTUXmGxn@O^>+wj+}yd&#-=9#-kvK%h^Tho0n+&s=kk8c-3>J!P9kwA9;w^8hLoi`nTh5
S7g#X<3Q%s2#g1TqHX?(sQ%|zk;Y9<CtUT7Kir?gb#A;-z4(zjKhJz&eACf03w9yuS*I9EOxf?f3P1Ez&tJYhf8Ll5k-wTOP<#6e^}AD37&l_1DU#55tKxls
%3{Z(KgOe=ss4R3>X%3O@h%qh$vbQPdlKhz_8%PDbYCOI?<0wZOyL`Y{v_aM=6Ro_p)%*!cqnPxx+qj(J|2<G)YnY{G2Tj9yMTF7wA!cI+v9ao+M!Qc%U_>T
w%@2+bt8X&d?5N?MHJ~1(chJyH|cfR=pO*I5W4=bjJNSEw75L#hVGBAz<db+_=t7;Vw}R;d@%7AbE@N@ahQJq00023qX^G;pb_Z)7k_pcR1aXVXyjSF`WnKu
_o9A!r|SGXL-pci``eiHH}yH8`~2$nJ4N-~!*#Cx71diTc4J)mzQiwi&ol0NB94!Gc=#Un#Pc5j0DxLCvZ$ML%rG9$7r+a;rvFU!V}nIkU!lfp<9&Xt6pz+S
Byp?(5D)5G-~MqzLSc)7p%|YWQMdiCwlT04<4%@WeO)f$=*ZI#2PUF@D`Cs}8OTDrXeoD0&m0_bM|NJJ$NBdcch7O^+u_Z*`;!rR+&_z4PiYD!^!>ELueena
BTe$-YH7+}BQ3`Sh56oP`(ZH=M*CIWpD2~5aAM5XTSB<jU)k>sZRPr$X&H&Lx~N>wT)SUd_o~$i&k;~14w|M6$vMc~UXp(Q!yg_1LWb%Y9l|&T)k=Cs?1_HC
x@+%j5o)_{iA$`;>wi&lXiE6@G?njTEm^-!|1|ue0RR9{FTqFsvoGw(#q&i?1h8QL3IL!!s9l2g`$4)9HF-~7D);?uBbFsH=U=;b;q6YJ!I+0d{azwIKhyde
jynhd004-Y`dZ?j+?Zv(XzM}kU*hn-0ssfh=S4_U48{up006)vec5<tqpR)tOZWGZwC4HWo%OS|Qe*XLR=n1gtmeKRI(}C}-|v5nR(SsHR%>RTaFxHl-+-fD
^y^up3MOUXeGY&R5o>!ap~lMz7$*V){vWigKf|7yhCN-5{seyu)sN_>cKO(reEm+U-&~dRrSWxwv{rtz1nMtp(q8i)n%{@s0nlzNziw>xb>p>KE9c7UzX1TM
54s<?8RHcIpmB`qT{o<EeX(ffo1w3Qy1Lcj9-l(byI$-2pDET0%UbBx&d<L$Y06xfxhF>Ud+c%G6T{I<Mk>sw*Rj}#r}in1r_$W2&A$u6yJDW_iIJ0|>h~y6
-+M64&qF3ZYnN3vBP}ZSkzyrvLau$Rxwtcf3^Q6Lp9<idKWYCo-v2iL>nW?c=VjD?rn!0My#0&*sjHRm-y!~^Hh1~0mte-B!@1uZPLL>`KWM+pw+p4etKSdd
^od>z^vB8fAK`xQyy1zR<1w#^ocb){^Xp>HBHZ<2+OBg2TRv0l7XcMGGHg`%zgLZd{AelE@7&00>t8=sUvHt#%GZ3`^A01NhO52)W-$_J8lk#>u^&C9iU0Tn
FZGu#XnA?G=;gxbq?<IUUoVQz%PBfHZm9kM^f7^U^4ffD`msNQNXS>I&Le-|zK>mYoqEfmc%0R(lk~pxkx-0B00cx)Ff&>&UwD4zCf;IBbsRJc^92wDG+uio
<gpYQzrE_;D^%Y_{(A`gUjd*P8qaG3#t8ub!zGiVF3ctE*Rv689B!dN{l4aKZa;Z;NkX$%8e}#tN-_xcoVpJ6%Qgpc000000H9OfH}JOu#7#`J?~bD#76?|3
t4se40002LP<;UHTUVogX(#L4b4=Oy<eIiKpBys?6Kp%t{Rki4%J-l5|0P}F{iy#>#32_uG_Kx2^dkTO06{_hm@2kwyi&M3er0F*_AAEb5+id7|L-f7xt<oe
p2D}Mruv5yb>=-_$@NgZJ-K7>-U9#t0B8X=EZd1V$~g;uu7Duke|xZT-=eF%zrAQc4ai*;q4A$&zn6#P*6n;bGhJ=%>M@&nr}umi`bpsZu4zm^`nvtOI6cUL
z%VH~b!OWC*@?bitg=6K-`BJ=<O={D=IxfJXZu9DU>*RXK%PDK+1-~9^tbomEXTvoY`dP3tk_?xaU({Wyoj>psZ96T&GK%g^;dXa+Y~0meS?qIj7J)0H45{K
lKekVO9KQH000080000X0O;NK8fO#$0Dy-B00{s907qYQb!{$gaCz;W33M0Ly~pJhmBO<Hi6knkrwSZU$_e(V;QKYS5Vg=2eZWJdG?D<df<OWaAs%=JN{kXP
Y$AyS0!Spm5SFkW!X_yoAq2u|*+Q{kA0S2G+4}zfxlCf_pJA5&EO+jFKj(iKGIQtN-*#uc^Shb9?%Vsf1M1x|_KxJX@d=58N4I^db=z)3o@m>#b=x7Yk4_pL
H)7E1qvI2T>wCt%mY5J&pExXTWJ2KYcF{lW+_CkOKWpEybxLb<G`s_NiVq}CkyGu>liFKG@)8~}MiYq=GFHA2b_OQm%s+8cPx|*H`UwEHp<(5skxw_QL_ZOP
<cNaUQ{!{tPr+!nB4^{2D|w{$x6HVEsve(NiSr%+z&%<@6yP}Iirqhl-2va%sXzSc;Z<`^(y&fBn^LbW)QnfX<wOT|5R1g-pN(q%nfd4S((a>{BOU+{0ImMq
uWyip-1hhWjWat;EnQ5;%I~1sf6pSj+KuT|(E5@7TYPuCG;3D?fO!g+bvb){=p6Jr01WW{Po53@wvw~$twcXn;ep(kz;7NsC@qhYTkY#Dr?)%+{8NTFHpQpH
*e#-KVbiXK2YeR?Oz$G{rrg`mZ#@bd&t3}u0UN;hEC2w&@zkI6j2xJ>zsFkN`7J7_cH{|R^=99h%^<7ym%lGn+iq98*4|GT|2|fF-QTz7{kYX~YSm}9J$?1&
kUQ~dzTJ_nN>b{7qpsJ)XX-t4=hS#Ix_;W`CvToRzuR(qu=-PV%VSip+S`-Js9akv)+@Su&s$_puOw2+CD$vjUodadt9Rw4k(wQ>)knL0-SV3MTdRH9zFP7b
)fa@v>n|sFzVy$&ur~mJTZ)R4UwHJt_E1W{A6?t^L2s(}+|M67-#mwcf3=u6n*Q4L^~ni|E2{r4Nbgmdac<nL@&l`va((jOUu=1&xL@=Tdg<+`=YNwqgH~_6
G-k~Y7v5SwHsrXd+^_cf-XZH{2LJ2ZmcKhtGY(*B<6#XaEtwc1mvaBNr0ws8l#}<b-Tv@f<x*6S=SscC9&6uz{eV<DC)qE9-csye+kV3p%ig!_o)IG7wqfH=
U3e|C+K#eZu}dz*e*N!LRqWb3Wc>=sk7=abUbo$9+L85;_fzhFu9n=xLzc4L<$Bq2N_9QWM&%FbxFuclyeXHDl*)ft9cOZ!D$6PDR&CFVCmTGJ|AuYnX}Ny3
{1a)FGrL{pYTNtMU!9wU>keQleyoOl@34ovcw};I^jtsjs@wno;8)oA^V2wP1K?qP_lsZ#uI=c@oDkop(STjuXTu+k+ItPzTmKS5oB-f`*tnP@=yw1BRiY)%
;5gyy^TayxtNdN0_&<R72afR0l<Q{>RizO01^nRk@A=Kj|0&|qCw5btr^)N{H(}l~*w0D3IF-Kve(`rd>C1UOUI2jqm@_-bIUJXOr@_3^z{%@*d*$cv{e@d{
x!W%C`uPjwhXTMmTta^ZwP4TLst^}kTrXn0CIEncoSHv?wS)1g00106Avex5z<v7Rz8RPxIieu;)c9QTs~&8>Rphoj;mXnO<&3TWXm=d_-y)ozm)4wo<?y#h
Ve{7Jh4^p!U4IDt${d^5*6MadnUB}#Jj8mp8!a!2A9y{U@_w8L{)FXsU5>bfy-ms8Qd*Du0RZ@63x!)gPvWR%J!~A!YV>~qfbU?}KNoTQa~XLa&M!^`*$aOF
0Dwne`LYj@JMAvP`NHG&IJRbZ;I|uo)CqeZv63L4ShUqSK0_WiqBed+@3<1X#<lQ{DC3!cKM{32vy$w?X<3KUP7v%1fcUhmh-py~qut+INaeCfUaD=^S5Lm~
_C&Y8U-a~=zM|acNskMT9fIFD=jxp)F@wtH==ReUseC0FtAA-o`8T$=7!`Y5JMS+mSAsa=GH#ZVTX6{71$kxsNpZ``7y1+z82lbCS0&%Sc0S46kxw_QTvX<n
IAi+GB%}5K0Jx1--cii!)H8Bm(*7R$ulu5c#Un$X7a6SwAm02y)Refuub8Z<=VOwWYTM1#S4`IGyNis<6Rtkt?q}8hAe{S8{_^u<-e0l&37kQc_xUTE1{FDl
I0gU!M{(-;OoedIiI8XF_Q?4;*m6E;<ry)Whs3G8C9?dy)(89r=3@o`xCMK^u+}%npM=cbKRXh?%`tr&w3=|K<nE8isruOV6=Ze$%6jz4@sV%OYpt(OYIS{*
b^5dVGRUd+%s!7~a_l3$avRlGOHbu?z3uC(HzRDljOrUTJ|}W~4#o-a2J)5z0002?GiHB}z?~}ID?0`9wmZw&C#IbV{9L#9?BoaH)-9a@zk3w;&zrFKKs$&J
IgoFi&-j2}fHUCvmOfZs@Ir#z)*^jI5RG}G0pN+z(t`m!hp`5HPYu4$X~ct|AYYIeuyHYq;73z-3ms1cd-4Ik<G0sdyx2GUuj2y0&k1n@g77~6o4xnnPsa4Y
cuC+boO!=fhJFDIP{C=u7|p-6z$HDD?V6PX&j|srgWo)Oe$}L3f2g~EJvRH>S4ul+&+EYIP1l(JZk5h@Ht23o6u)>qPqa~gsFgPwp8yzvy_YCOd;x(tiFgJ8
cpmwAHlAw(0K&!S`!oQoVEW{a@ogHZkADkn9I&hV67J&Cs9ehOcIm~dz3k#<)clN+8*_!x_g57Kzx?XC3f~=v&H7O^{{tWrb$`E(=WqZ3APQEE_ikK2Ch(h1
R+R$)c!k60&%hHrC8gy<y!wUI#@!pVNnJql(&+v>qXxg*d8hjLbhGvHY9~Yt?D&1yc*14Gr(C6^l#f?=MMbu^SM8oB@n?(V-lADgKk2#Z3B)H5nszI4Hcq*c
M}BSJEaUE}dVFRjzU%cGU8MY$2jDm0DOg_MmBdRu000000BXdoTiRrHa`?|}l=JJbzbBDex!Z4_q$y9&|Ni?QQmYC=>e0J@czPPI->UhC?AFh2ylw|}ef~a~
+wXru_IpO!5c=JbT6w+o8{I$L_JKEH-w#Z!_H+Kgw4<{=-b-x#o1|7wU%hXV(f0N3r?=gW_pa{HXM=iwmD^XAQ<Fn3XVe~6+bvu>+a+(f{tMS0Kn&PCt9#&g
U@Po>dm2F=0001hAHI|Li{t0R78Nmp-zLwWhnXH(#D^UD#Ft&P<#F47YVGr8PrJF=ZQs}3_W5t#Tg)Q_djw&5Bo>o1?W`tV^<w1@qCW@@p8o+6sNZVBrINdy
zkg{UwY!3w{=?+?2>luM5#YSE5q`pX8h~5x>rNs5O|gZbF961Qs<3I-!luV4qHFL!iyVt9r`z+p9%^#3^+h<I)M=M(u!~UsM4ST<4&Of7zDxYIF~pV$6j|8w
zNb6vrMk-{ka9m`?Ic*<U%NHP&7bohnWSt-tv#*nR>_}ox@FtlZ%!m-dwT1U_t#hN7*hICOAn*|(`$#i<0Le0<n_tpqGSG<n6B79ne?_R*QaK$y6sk@_dPd?
b4OjDUHy0N@tfHrKL3!ejC%IWrAObSr@tvb8gXzxy)^CaFTd_Tn9};DpZMwrDKz!gQms9?JX>bc*!$kdOL}PGt-ljP_LJL__fy-hT>rJSzGj|}O@m+k{r}E=
N7Jve9Blc5Vp$`fy#A4i?S@?0I$3wT$nwhjDfMFZ8&2|4so%rWezQXMtNh!MHX{drP+ea3$MpH<{`ilJq5fC1CtLq^wccua$nweMW&74%U$g&<ih67N^#!Wk
?%VX@t@zOLvl^QvdFIh&X}_te`aB=I{KbnSGH|^Fgo1sKT29E1BPMJf*R{g<H7cl1B=S>Q1>U>5VQ=;Kx@`M$3a|d<@Mo>(T;}s0K9Cy|_|@d;WPi_d&Axtb
pI13wx_)r(Ge2HsxV@okFMZN^#t3q&{CjCve_nNQ3K^|eRAf77*@G8*nUzDYo>tRDqJ+F!f9-Y(h3PM%=>z;7k8FSMj~_$7tc$<F9sqEAzr6_kQ-$oqX<3KU
_7ZRQgx`SoIEi=#Lc!j{E+pRCJ-TN2vA{2Exs1+xHL_JnO8swgsP^BXd9JnU?KQM>){dl|rsY>_*DB1`Z}&W6T5>73+hr%K>s7NGTfZLX6&3dG333+p0j_ZM
&XkxzWlO2<^>avGTGP%P-^clg*yBy!*sg!Pjb6K{`B7~<{J~)JbL6;ghi9dEV!pt60D#+2+;Z}TKE<0g`$=*{LF}pVtI(f>26+nLHvrtntKS7Mxho0w@&cz#
+6%{90GvCnEk%FR;C@cNI}1uNFbR3_0PqqlpXD<0YW;lLv()qYvgdxe<ac|UwS(LCQ?{@7c`vVV<dt~z3wF;><J9~&yvv^_ahgQc&+C2$&cv17^HkY2zD}<9
_T&2U5dQ##^!F-!FV6{1+ljTOJ)TT&IRJqF7=!l`065Hf^f#ZPHh(=Y_33LbpWK#1R0d4g(SO2@nsIWlD*(VL;@dPDu&eu$kmJoO()x2a4!P_HzvI-nM_$H_
`)!;j#<dzFwQ=_bZBom~+WHodn05-)OH_URs;B4#uL_#2#CJBhP7)I~o~zt*<GC<S%Ny5;?0wgMVeTJ9xn9O~zcGe&%Gs29Z6WOC5r44lE6FN<vhwAGI1mf3
=JR%1?oXQr8^pL3XTWoRz$YxP;tWzQ!>u<~%9Hpx>1*Hk_*%R3L2A_>*w|!ta?N)YR@qI>KYHabTOTj^HG!B!O^FNqK30t{mWO{U;tv1-9Kz`PAJN_yF?|l<
yu%rGe~&E&=SfsMv-kA+$K|NK*VlWzV&!&f?vMB7000005EINV>ut*5Kj3S+L<c#8e&8~^<-ZD7o@m4s;0RIP+bi{8f2;S?XLOD)0ethF1dgX}gYR!}eI?R$
9`H77-i>vLCr9p+4{&ein^E&KN^Z<0z3aV#Rr0fXJ|?H+Qf}XH@9GYHHk4bWM=^QT4|=I5{cHR=d-Kmm1%C4_lT&YfR+r-qS#Gb&uh$=D^%Ko502pD9#6`qE
5D@;}_p>}UN73H^0FEK%bMTXxM-h19yl)Ke<#B8&QGkA=!GUrK*vaL%v8!EnKAGjXK5n(^t?M6t>|^b9hyCNC**x+y9kU;6Z#&r~zgd0Qez~Nt+$!$+qn6%v
+exp!>h%`4oP42AvHs(&Rq^I@+yXv{_Q-euU@vC#QF=B1rdhHNr)3>Z+l&4JfDK5)06*|fp4ySEN>b{76PjO<t=IpZ1}h)_c-p7ZzwEfAXcvx82K^<fe7x&x
mafyzJTdJI=7k0T0Nx3AKE2H6pKHH(X87-im>ko|DtYzRXSF@7mWP-6t&&f<^r}93(aN{Yr@X}kyFm}&9A3%0e{9Y0;E%i}Hvrc4({jYWFZ{gHv|rhAayRiN
U$}blYHzQCaj4$1gK|79lY6r;{%f})XXBJBc{rchWDl$A8+T9D<1;Jm|879EcwqT#N_FeQ#sO|df8Z3o<2}8l&nog}zdAx}xs1ZA$A9^z!nGUh0KQL)ikQGJ
i&{y`6I6TsCRO`38+7@;L6@zchl;hEPkyt%7<zw-{s{l*fdBE0zx>Y&cfD{{|92HzBW^5zuvdv@`LgvTfs!8*TYf@y*Hfsoo=?!<YS|;*b$c}LHDllU67~BG
qjIX1m*WVY4-c(zJZ8uF?2H=xZs(oF@DBjO-hUn;XWJ_k_IRRvUk(89FP(BWrCz(&xqOntqy`t__yYg{00;->$p#|Dm)G96StT#;d1Ub(ls(T0y`TD*G=Jzm
vTGc-TDyL=ckS)sjGJfQO>EGuPtVAKN&9=O#rYoq0Dz?o9TMaa&M&|b{5JlX&1bosAh#I)WAg;;!0{3Q%P8aS`%}LFd)aFr_8oH>?5kk)svu`!7eP_Wci~n(
i3f6H0zduzn9VOy#BF>seddw8)T^hmywmo*Un=)!;>CX9?k~rQ(fD%OKfLjOM_x&|-ZPbWsR#HIHvju+-SLC_e*pN$>F=BXsEhbEjRx%MzJ#E^y#!lnwle(j
b$<630001cVDFPJ;{0MM7`F}pu=+T?hy0=EMuU2WelLdcVj{$WSN(_a6#xJLKwXG3??AM~8G;`G@C(R)0{{R3I0+l?x|;mjzST%+q`v!$ZTc9UFOZerPT|(8
H_tofn*_jr$33z9g6}^>f#u^n27iJ&$-eh&`<U$aEsu}CKX_<!+*iBF8u`S{8)}#RMU@KxD`BU{bHc{Cx$$oR0Qc^{s?c8n0C+y2AR?zhk9OS)b;}zp-Q*qZ
@zWzGHG28vzs(`F`uD{BC~@vK?dz!Q%kK{Dc4nQ<_Wp$9yhF(TwYI)c|FQVVA*cM@zj0=Vsim{XuJTs#tAf^#^lyB?W;-k<yX|K;diBtf+pImTX&3KxybjA_
nN5(}4*B>UA0SOf^*-{}0&|EqpBbo)+uwKS9fyVf<xahB^?y;0JNEi_tMB>^#xnttxfwM-qvXb1`eXlvV;_FCl8o+WwH!{j3jhe8VV!a|rC#H2oE?)3_me>E
>vM10Bi#E6t@pFYe+7Ux$Rh#3_h}-LL(mi44)&bzAcbph-u&S8d}6>Fke3Yr00000_cs7|4feU(A#URs?}q>Yz)w^Ndc5D_)AN|o`8N5TpOanpFLq2mmj74#
eS_0-0WsnW0002cLqylYrd<oSk-l<8WOlo~@)wae>|8|timM`A_W%F@00MBJT!Q=aI?E$fK#&i(LC?s6N&9;&CAGiZ;>(5UkF7s{8gXY5wRyP`rHKSiS!P!@
FTs5Hi4)xNNAm@bJ-Pak35WyU;z9DuYXm(20A7d9?>iI60pJN*RKx^+SycD>_jJc`vq6{d8+2Ja|DjgBY<oWGYj1Jb?RgEodV|{hKTt~p1QY-O00;m803iS$
J|tf%3;+P2M*sj00001OWo=?*a&u^Da4v3edCgmGPaD}4z6{p}5*x4`7GnaY!GU!kyCx105*iW;q)P%J5Q6P2SW!1sD^=Ypsrserujn5xmd=$PJw0b;>?pg7
G%_CFxi9DCIj{5I<MX4F_K^FR`}gusA3y);)AHkm<?X*dTwYyR{`1eD{`u*{-@8A5`sriu{=tX8eg1fT|MOoy{Nv;G_irDruRmC=uP*#&VbH%DL(aMGSH7ZO
V2yq+_Z!;FFF~I-13owPE72sQ+{-3BGx8hY<>-LVdHGyXej8Knts3%t<13u$CVfAI|6j;wPI)e`oUeBs%jdYU-fQ@O0?yZg`;+py3HLVol#3`OFQ@3t9`2W~
uY!Cp!tZxtFvrVYODS~D$@q+dS(n$>ExI>MdJ=v=fw?`MV^-ncH`iDD`kEdPFUtzOW<aki*VjdY-aFYFe!p$3SCaWEay>J~HR1m`_&-N<XRMirJrw((Ly)}%
qDPM8A^cWV<`|aG`!vrkeEtsfO2ZK+Yie>?mERWWeog3*2YvQ3S8q=*udf|Nmyl=o<?}AwUlLk}?+t@?kAQ}OAIq|b9*_57?3V0z1hC*+WzLI~egS^Y5Fb3r
Y43X-Z0tRe|GO~eHpo{+_;U<cz9s*EFEn$6qiw)FPdw7>0m~ZlkL^fPk?EF<jbtxZvj2*L2VVA9iNV!3z)!Xfy=zZ-<~z~rd;-SfnJRML%>jKrmtac*R(zi@
&#}z)4CXC>tqAk)_kkrP(ccX^`y<l*G?+XtG;h$^^JL2`8s%XhS<yR3n6?r##~OVf2mK1q*GSKLYd%QupO{Zg#Xngz{1wZ8DS-V9*y9I?F+AyEZ=Q$n`?}np
gYhieY%AZvI!{5iN`}sC&^~Zqz4eb28^H7!?n7n~bLw!moWiTFa@NS;@wTzvAL0Mam{_7~*jXI=0QT(_{0#hA18ge;mRE${73F-7L3i?sz0_=}7kSPbIC@Mn
;%QFThwda@74nl@!aOXyb&}&8?ZpicKRp&ZGbOxqq(dV>udG4CEt0#aPh9O?Pn0oRimdM`^6W`x%Q3Nx6<>`Qa_`CiT~qdXpv<37(4!2DDH(G<qPgl6>-7BB
-vi$@{az6)&M6rDlyLV&Onm85%2NZz2Kn8gF^=M?0<AHeuwL*FdcO3l6tQVe)|obbA2;F%mx2~~!#B&w{?h2-u0lhU@r>LXm+voST-Csjg2+k%c*=WuPS}sW
M<lX!hj4yfaB)>=dMtRq7b6du7>ugY_mYesrZ_w=<2aso2Cr`@@%rw7_BULwjPtoK$c=6%{GL<h%^Gr^1G%dR-#Mm_l=%5I<b}E^^kBoFS2jjZJ?;4qG4YG$
OKy`r>MAk02ifNsE(h4XZ}9QR&~vZE9QO<?k7DdXL*)7Z=By}k8ph5kah`*7d*aK8!5{v<sGKwDnSSLKeL<e9q4+aCA;%ag_~B#ri*&L0?I|CdMt{d4cbnt`
I%Tg;aMh>u|I*}N8!2<#oF~0w%wIQR*$C*|gmpaFc~7~EOv>|l#V*_dA1CO_ZbDq_m6&Erp#jtIMGT#H#qP0vwa&7x_!a%|^9RAJHL!bIfMHdTS4ZoFIQSUm
j)bT80C(R&jD_~ABKsd9*(oSK>S9c8q9D&E<%7~-P7Y*og=|AUg`ZlWxew_K`^1YwCGU_{H*<!r6%5?9{B|D`Z*G&VdZ~;n!2N)oH^t6x7`A?$d@aoPR}$(n
X)$by<Y)_g%SsAAfNAUrBf2qlwKzWx{m&eoG2|SMVrr?=kT=-?{mBV0#z7Y9Lh~Zgt&}om#yZ)aQ3u*Ia;w@JHQ1ZQ#}lO%>8a+Jbk&Tyc&|P=6yx`GsV2~(
@ry!B2Yanj4zAOujeQ5X%MdG70juALEZinvxwo%V(5H;zyFZZ}J=e(UiXV<VT_O8^E}v(@i(Y-`t-|x1QrF}<aYN4b9pqsm;(yDKt8=h7PoM@>GW_Uw^jlNr
Ka)CBM73nE;I@ZntQys*>{;sr*nxcoKf5j1(4l9%QTx>Vqem%wNy-Vn^b)LnKzQ8a)q9Zd38}SN{_6qN!sciXKPu0h5{{nr*>5wi*y2u1{Dis1vQme3LcjM?
zt?PiHo=CTlYZwZucyh-W8g(s>Um+0+hoV4sQ-awp`Adp1LOXVg2fKjnuUBthIIWzp@&PbuU)0*q^r~iu4Iik+rJ35ZO6djW$^1K6?$YzW)1+yvZCWG4>hsZ
WAgtiun`Zbw<FMMh%lxA@?TS)vHXlHk*_Y<u1lpi2mOZ(<hW2?vLbIxqYTI#*9?!S4pL5#y<&{Nc}2RPHR8oG@h*~kV<5*Zs?Br_d!)s&bMpKo$kB{ElNJ5`
hG<k28eIte9oV{>k{8S<7}YTJrV2QTInr6VMisgj0B6pjK6V29-U9!jEo<;RGfEs3{tksFvqF~%vgcu~Ly`FqzmF2epd4s%_uCj=H4UF*7Wh(wbt2)<Dfo6x
B}aidHfP}c1&l2du41izlJb%vPAbc|=iqPtC^X9eo%7(YL`EKOi1t1Tc+g6y`IQKJibf2UBVYGaaQjHssFD1&0h=dejy9}y*XUau7h8A;`rLp%ol83Jt(5Wd
6OeC5cCV<=G4%V)rtBrSB){rJ<~~uzTmIra_)QZ^uj3i;E(17&{$7Q0Lpvga7sBf;!yY@3(YLb47We@U^3NXdzX@kVUpesGk#CWO9y81d4H11>^8Aj%vmwLB
jbgAPAJa!#g7~`x=Sj-@91;fY$M`-G&3QpG6Xd4eN6)p@i@?_z@a(qI^B9+3YRa>dfF*O34{VUnHfF?_NW)X`+os76EGG21zgKW$n&@;6vecnDgUn8Y?-Baa
t_bsUKwqcCE)mc^Z}{5_P-|?%JPpwG9LPvnIXnBGZIXBNDIdaquH^g?`7FU#EyBLqq7PGGZ!dt?d8MadDkcWQoZSWFg}uV>G1%h{jhm0DVI$7fs2+G1<e(1m
X%XUt;3Le!{8hoxn($>w;;^FdW<q4BL;E?Fa~uku%nJrZ5QF5!=bID%y&-etslIuf&=Y$KbdSrcm8pN_HTY3=*<;bb3adY;3VutF%~ko#gHCt@2ipohL?Gjy
WVHqMsiF8`m$Ke1CFj>k(G!Ha))BeMLtdm{^tzn^evB%=7YVy6O71M^o>$`e*A#Pk#kNHW@1%H<R}V51F#Qa$<B;a_;G@lhUC4p04Dy#HOfD*Wy9B*ng5TCb
o>8XHWn7E;h&t4xcLuqWF~b+o?D1_`;~m909Yww>z~@tiPfdzVIto2I#6wT|-Uj>$b;_pbXc^)F4R6a}lOsCAic$M|PQJ!7*>hQ`v*kpFrevMBLU%{|2w0K>
`!)|cyhvxg5*csE*_={4&}!F{3H?$HrH>KoAA|H%BJMr~zAaH4HDT!Y4B=&viA}JjJ21WgIsZ$e7GE}c^c`XLnvx4cf8}P%vuiy_EqEu1V{&KY^tv(c!f?OZ
se*k@(eVR%H|0uX@s{A&4&|-9<a&c`nHM@WfG+PKS29Pr##c&wwg$fqQ%={)g<@agBKhc3gtc8IFA?q)$RA@HdJ<FrjuLt%R*igZnQ}=kp$C1MKCSikQ{+Ic
{6}2>bmj~6-V1G8eLi<T#d`<$72CTC{iI22Oei(bQOd`RCB(9i6}vZQ?D@Sxp9rv!d(Q3<K5)PDgM_%*+Gp3O59eWQ{R{ZcShbOI?6F`;jcm`6Q6DcE@#AVA
y*(wOH@=S)#prnxdOQk>-PjP`>?Q2W#mI+tT~dGetkK^;ld@O*9CCt-FlOET|9h)-<s79HHLJTyZP3AbZxi}Ttv-NxsB3QnzB)yBn~MFffeqyMU|LRp2YyLt
mh`?=)u1!yB<2lF_TUe<0JByo?-<vk*&>-a6pU(7{u1+S`?B{<d9F-zW)u32IES$cbMGn7MG5&d2ljnV_!xR^J@`;gd><!gEXLs9cAqv3Ysqcs``Z+|?Nh#g
cambvS_(YBOvp9F%{9oi6;nSRgpL}f6(9zif?PMhQ*fr_8N1|rqVJ($UPa{C8F<CIRU#i)JHwFc?>+jjb^Va=`~Lt?O9KQH000080000X0E7M09OwW500062
015yA0Ap`;Zgg`lZg6>>kIibrKoo^1TBF9FXpF^LDkBP`kfkV!Yj;YZbm2-ujkHiylG2U%2JZS2K7&u^890ks8~DhbJLlYU=YRFF4A++U6DPHe)6YcBq?&(^
R3O#vko+dmesf5+vF<OTpEO24{fdq;&V%VV7zXkzAOD=I5W)rTzymkX2XCMPYW!zF7EC}0s=U*lHs}~0?SBGA&;tc90MD#F>KfG9#=I6NaVLFefhNdP<8wxq
ys)36SNF<5GNWX?!DhBay~kO)uG8yqZ`a7l7`X+`^tn@;9+w$K<~w|E7re5zIjc_11)@7pO9KQH000080000X0KsJy_|*ac08Bjq015yA07o%ja&LDoZg6?+
+gDFiK>!6{5ClO4LBJB!Rg?u;HQ2!#3$}nQ_JXS{0UNj=CZfjPV(-1TsIj-GiM{s{doPJ4_7Yo+V=@nu$;{lDduOkRbCU1jod0lUCALp!-PzqO+bzeM?8xlx
w8mMib<(O?Z5C@vx--jZACQ#pOm_U)<L&)29bnH)wP!d$kE~iNHpUiZvkbH7jjx-IxMT7$cyRC(;6>mqK>-0i6uy!bMED^JN#IWqz)%<ngn}4~Pz0MKL{TV0
fntiHD2k#eim8eVD1iyXlth%$V1dd|lqCp<%3;c5A}Fjtq<|=(0z)*y235pVf+}l>;Z%j9swCAA)g`F`)f5nGP#i;ThPs^U5!5%Jfrf^XH6mziq9#zh3pC}>
Oi*(JS`f6<kid}WVy%ES2DQzl+F{yLbP&)H>V)VFb>Y;Npc{woP!CL!hMpYkm|hYj=MfGdMMEl-M$t#7z6SJ1q!VNy1~ND`WOB$dU=Tw#VlXr$m&(ayhW^Kf
LBpXD6eD#QWx!}bV=!YS8;2OrFaeq<!6d|FMG-VbQ4~c{lz+}tf@u`fHO!D?rio?|%;qphz+4XVp!u8@<N^zISo9YxHed<CQfL{)asev@ti-IMSgq3<3Dy#<
BUtZZ8z?p+Hc7Hsf-MAF1#QD@*RVstP64|N*ezg>$@XIQaoR87fB^>)hY*Ja9N~1-fMd{c%n27cX@XN6P7|CVIIGh+g7X?K7;w=fmnbeXT;X)ppldo^|4VKV
+$6Y#xXo~f!(E;33Aiulfr~sOc$7yxCU}B)3Oy6>9P>iJO9@^9uNmG@yoKIrc(2n36Mcj}arlh+Lh+U28}J?aA>b$QOT(X&UOom7ojeiw5ncv)6BGb^AYVkm
JjTytg(&<v1Rx4C1Oh=46hQ<dLZDEfsGwpDVL(ZW(i|)jgfm1iSPh7Dp(so=V3XuGP)h>@6aWAK2mk;8ApoptqfSE|002xq000UA002iZUt@1<E^csnott^M
mGk=l*Phzr+I#Kfpg{vFwaZXaon&^%331BQAc|5n;h=#+BRPd5<y4erg(j4tQ>KJesGLffNy!vt_`crfey^)_^*ew4T-WQ;UTZy{=UJcOp5FI*;DYvTFS<M{
aZ6%gy>8tHT+_c^%UbnX^=w+NqE@|Y`}Du5f7ctk^y%NNyIw!1YwrQwd3`{Su6?`ncl{>KTbxjFY(=fXwf=wqE1H*`NW2QS<s=e=;M&|oqG_c>;zc+iKapqy
?O_0nDNQ8GQ;9^&!bD<1Ng{E3GLd)!MrI`vySOg5Ad#q2#QVH=88j_RB<Atn6^wr??+t^qiW7-LdG7<R`<?NAV!Y0LW~-lJC%lxGNHpjE=ffwJ;u#jep^Vpp
&o<+;4J*ZSeadT}Bp8R+JM;QsTtAm*X~r05^L}Ue7xz7y=Q^4D4&}b(-1j-Cqif+=o}nw(p2js(`0Q@J-{pM%>tuZ94#?s&m%}IUzQ_6lmhqYHjI{zr@*Fqw
9E<dQ-~q;5483@!$?!PLfK9L&e&9L3;-0JcE{F4+Z^KybKNs4UGY+3=!uwzH{?D2>?zNg{>A={(!V<>b2`%}|cBok?=Iv?Nq`70f%{=RWc(#RHSH`oaU>_tH
dlX}y&6t1Y{R^NSW1h$Bvl;6Lt}o^}?%?%rxi-bWf8eqnyf=e?uVCECT)!4>W{kRg?uJyXMIY|_H1D6vdtbmp-)}xF;X4;G#^F#O=ECRtoBLe|MO-@@YO)4>
7-Jxu!5F_mUFQ2@_zk{K#<LBFwXlq5?9ZGX#&@Uzh0IGa918W|M7R`s!a%qaM#5y64Y@p9DKvz?Kqt5Z#=r!46Sl)Q@GZ3Bdz=pELObXKBVZJahevZ`ALj7g
@}MK!s<npAdd+L~AymycXI<fH7zj(@KX4iI-4Pyx4e%3GFhBiZ6ikBYun1l%jJ@6lI>Q6-Fgyy=U^>i!t?(u6hAieb2~FW4m;g_}diV&whaZb$57uVxkA@C#
9rT3WFc$s+(_toTfFEH${0il)S0iW)U6Nrt2HSotf;I3t{0up)XA*9Q39uFRLTlFe9r)Gfv7XK?^i9PXehuD%weT4n#(ro4_rbXGunn!*id_@2Pj|t0ko2B8
1{y#^I3Ak7sn7;4f-9jr^nmN37xaN)Fd80#nJ^a?!g5#xn{5xa!FKo>cEk6u53<>R1yBM>D1#JK_WnK6d$<uC=e^tn&VqB`TsR-vK?mprUEpfC7Ov0O^Fv`c
jDpcH7RJNVunbn&AG{AAz-IU`JM6P;&A0F!?1w)fhdE99?36)eI0UNX#C#tGb)lZm)``#(F7+Af2G>E4oUoJq;AZ=xp)e92fl2VFZK&+xN_YcS!@KYSY=lo_
SK$}fujk<m9u0Nn*WhGmo*U=4HCzGRa>MWShe2?+&+>Gb2T#I$SPV<y1$Yf!hqdrNY=X~VXKvW_eURXs7v_b%s}8lGc3$`<+3Zu`R5%kZgiGKu=mcHh>b$VG
z2G(&40pjjFcBufRCvO6bO9`c<$2*lWozD&Er%arAN&Tn*qlS5Ivfdg>}xwh7w7>u!zdUHW8oh#9v*;cFc+3OPFMje;Z;}-Z^C=_L0jN!*a^SD5&2>7szG(A
1+}3LR6r9r37SL8{P5xDK|APZf1b9H*Fo?6@HdK2ZnNDSFP{%nVIDjQ&%z2=1>0aB<Y7nip{yYM+mTQcYC#=n1kDN#*x&BZ1A4<?7z%e4M0_zF=D>6CI;?^9
w%6Z5CG7GcPz|cXG0?Cu?06$M0ZxH7&=xwv)o{0M{C)6{ZT%FO3UlB|SYX@#Jgk6~@F9EzJK!_;26h!jZ1WRjWA}?83FYu7I0C8{h5u*)XTsU`B^MM$OeCLj
1KbFMVJHlP5%3^Pgh}={@-<Ju4EvlX;i;mC4Hm-du%Rg8sZB)@t89bq@Ui{VF8B_9g1zuF{0e_S0)JErb)bp;*D1vjcbpD?gAQ;xTmc>7D!3l{!brFm#=#_b
6sExAFwH)21}uQ5;AwaUo`vV(MR*BbhLx}ew!){5dv?OEgM4QlXjT&OV=Fk%zO*BBhHe=?wIB3{Ti`Aj4rA<VABIUV6{f*lcnY3|<?tfB3~TL+--E5N%l<eY
KU)YT$?(lpk`Z6lg$B?ND(tVDK`UqtXTn*@h#T8L2j~jdz_rPUuSdZH@F4sXCc-3m%D#RnEQc3i1+0S|@CkgCjJSI@?1f+856Hv+=R*~!0rjB?w18G{I-CK2
g$|_=lMjJAVVGib7z_8qbeI9N;VD=Iufm(~Hmrm7un9Ih{`eR^*E0}%6hUoh02O5smz)Y`LOZw_Zihi|KTLqf%OVC}tb7BO!wT4)5zi<-uLDOx1vH0qp&hh$
tkWH?g`RLd41hafDBKHUVIn*Zb5jv}FLr$N96S$i!kUa&X+3O!EwIC})7S6={Nj9JKm4vZzC7Zps!$D%FOOX11{eT0!QbH?7~{C>L3kJ@z<I<wle5A{%g4%(
^m44bJU`Y-^LR>m<W7n^e##B|uej;(?C_cW%EC_<aebMQeLUB>nf9L6GyUC$=f}85Dz49syi&epJfEqRir19y9M0c0^$gs98m}$PkNiYF;ugmLg|Uz1?~A$q
4QQ1gx$pVBz8QXoce%b<S@<61XiIa#*VKm>3nE6Z@R`5O`)^$;a-d@wdpsnZ2kDtEcD|z+a1u<+jrdG)*b)8?%JFuVhQFUuDdNPB{Mp_1`@JfKpVWRG!MM!}
!&V<=dvu-c!h=4adZs;$u`VaZdEI_YXGAe%7yiBumfPpO#e3!0xm#c=w6o1=VEZ9EFv#bwJI}B(6}f6Uw@+n74m!~>r}D!F@T}t(%}J5t1kJnl&Apzd<wY@1
$_3@iwbl(ChcAJ>yml3zna^i7!X~(g=XnKg)x2dzoIM7rDvryGd|bJB7T>EE*S6-G$xx)a0M~Tm9t)gzUgEew_j$YUz?$_bk32*5iT;c|)A3R|f63vsQ<&e^
;5c4CoqP6&`|=~lK7o1sm3tLt#U35tcxGa0<PEL4u8ZPe$DW$A>l`1dws58E3gxaV99a_QRr}|ApM^7&i}RVCC1H<rW={6Fk7v{?G@j<=G@i3&evJJE9910S
R`p!o$rv>l<2&fby#|&?jJ_)`)?52kcB8V_M{|E0&$Gz2lG5B*gQ<?aPv^Qh{ywq;I?F{~mwk?926F$|upQ>eZ*h+wb>7QkeUul-Cd>Au>oFf>+^e!{Fem26
xQdhN=xjK4SXvz8-{iG#X<wr6r+urs#tvwh7xj~y81IfsQAbgXdr4NTf%Zu^SX&fxb_u-i{n$urS{`GnF87Vkxbmo{y>C<}P+dnhMDg3#MG=$y*N4l`rJ{~f
FF$<rl(MMbw8@FHdYjMdyyCDshwxs#@_0{mg`I^FTfA8k{%)6QN6wd3d)t^5d-+(Nt7&%Rdz#xn^WA&H&AG9sR7=|i$2jI!JYQfNsJL49mJL5T74<xw(R$AJ
+Ss->DTtW4J#>c}eAjp3Sf0HZyzN?0^;E1u?d+&CDOWtwIhxj_hV96i&XbbPVW#AVeV2`0<a=EP(_j|&X~5XGmW0jEVlEGdPRTge$9oT`Zg!FDP%p##u*Eeq
`N{6tF+a*j<+o}vc4s(8F&pE|;B$?6{wZ1EU;e>;vt936==*MWjZSO%dMaYxYW#h2cGO&T{b78Mm&>Cz@E-R%D>w44b@>q!PgRUn8ZnE`|HrwJPwvi-b=R8f
EVfeaUl8_gFV{V$7?o>txi+gj>a&Vrbbh)q{*%1^b1L?R{PW|k9dyZ!{8D*TRo>sKT4rv{?}xH!@U6yVtVxi^*lT&7#yroWlJGqjXUAH;;q{yh|FQ4YIl2Nm
G2RBRm+ZRMs75l@WJoG(%+0RR?02mtZEF-?yp<icPBn?Pwg+``WB#UQ<o+9SBhQ{=8~9wM19jUP`C$*++19SJT}=B9*%8^u#kM1w*RDM4BPHRFWK*WwX32hT
RQ;<g?5J$0)=Rm7&d5gj_1xIUtMg($bq=&2l!M(@9BZK3%(}d&+3Spbo{CykbJy5a=XkkNoJZx8%EitrIS{ieM%B7$ZLi_;cd2&9c)xN@KHu$RSPY}M{t>Pn
>ss*nT>D%y_T<Hi9~t`wp5aEm!-cMcHFFNG?{`&B*jnW@?R9<`zY*_U#A{pNHsyZ2*VwV&BA#J>e&`85!(X{qFXbQn`=9U;pIy!07jo}jsi;ZMfGf(wzpFOZ
*1k>aJS#73zt-|f`xdR?dyW?ucnt>d`Dd(g{J=d{Dc_L|NJZUWaqUUDz6bxmo!4&izWWkNU8AexnwXy9KG(6*KB!s7zT21^^`Xo14(z*9+pT(ze`di3c*y5q
ICC`;2Jn2PwnMw2aas74>d=K}c?M?jEMLPd-2XxD)j{<r#(GEgK=GLD3cMj-rFsVA_fU<1HEs<L!e-b4jk(`9@GYFe{n|q>n9qHGfS+JLlrUEvQ=$3%3J>u-
GZIlteH5m{1%9sjJXd$<1JA)K+ko1x+x;EJz&Ov9a)iuUTOFuyjqQBd47e0J!%Ww}RI5^~Rkt8~O<TACM#Cgn3ST<z9|1GGC%=S46&pZpxTPr0!0(=)5is30
Jcl`H3o~Gc&*|UFVsEPcuY6BuNBi?r#gIke13p*n-u8DcbM$If)QIKd6l+vbobT9Jwy=inUs}siz51V7QCAx6ddycvQ4@KiIP#An@F`^T`69;-hueo-?OO5O
jx(xTN02_yIT?N=od*u*KG*Wz+|pRr`r2bW(@|Vo#_Jv68oduEa{V1}JNNwz#xvGYTvy@emaQ0F7Uy;#EQLmS5%bH>t2VyF8j|X%dT*L@0G$o(bH#X#`P{wP
5xdU!IoBCde5bh@UL3aaG3Ntyl3}})6Rax>KX{zJJI`~3HG>w`12ng{!wAPhl|5hmttqLl{*ZGu#cs3U9BW02k0!cS+}!os`L5Ge(%FW~T#J1HzIEO75T3gM
G<41L7Wo$#;u@rCkBec6Ymlp4fBXRQTwCn#8lv{Ga<!e-idxC`ySCTYbvo7M-h&TZll#iGxbNXd$YJhsU0dtw+SxqU$zF0TEK@g9O=}R0ga_bJ*Qwt19{kQ4
k?Kt+Dc*6NNp+<=T~qp(Ye=deNq3bEe#f<=PhC4w-DpoT^rP2Z%lQ!UT(_y3QJb0P8q6ztkG*vWRB;WaHJs+UifSe+Tr+vg^@R6fXX-#5L^1S9)|*uGkWbiQ
ZAr0d)eMbEd#Ek6&k8NKQ&wo;(v^BzQ&P+_SUI2f*(27Iq$O>!j`WlDBiV&Q>qnUyk>c}a_B(B@4|TBydV@517?hz2jqv^*XH7`^cb>JOCDwzsx~{O(`HA$P
J=TMi3;gMS=sn6k{%W6h0bB{4?SnJ*pTX9A(wgfd&eNos&$SO&Z7p^UyaQ{k`+RL}Rypln$J<%Cq5mA>m`*y-iPnEw*pEoV?UbPb^{@^k|1!{ekm^*ASkILn
v>-QZ@T<9zcfSMg!g}Z6-&g~ZjXEqZ>P5Bfla9)Z98tRdNzfug2T~q>7PNu3_WeB%(tn0!Xg+D1_B6Z#YivgpBX5CE)jN<EdABs6{drL@OUa&C15(YYcBRPE
)Kl>{+n!F)wNk|RsvGr%elQH~wH_obXo_Q!nbw3BREpR*Qx{rgZAiM%2KRJ)W?kr8+cw3~$`k842fxyKPuGLAoe3G*&P>NZb2D_F1+FDZ>se{NN3q<F485no
dQWBRJ$0<}oC2piuSnOnr1OmZANtN)8T!tLutl*aY=`}>lck`lb)K}wqncaK3@t~x{S(%4Ub1#0+q?-rwss>s{d<NVmd&oPMsx1}&}X_?qmec<-nxwZ|4iHV
1sNL5v#>0~-lw&gZPsAY+KYMw@*v;60`=fHYc3~SYq{L|N<Zr<(oSYuH<4zt!u<qkpKwUY0ZpWpb&#tL(m-x@oTB_-y#2<s46S2HhSu?VN#sInGkl7A5DwNj
j&v`=@zylXu&<Gxq1fgsYZ}rsZiIpERk$0*9i(kM?7F6W)0AZ7+44~f?Wa~)&-l@LhWZ<RPe#mD#X3e~YZ=X~XPjXTV~{lr<#4K*O?2(_N%tGAw14}+8pV#%
12vmppsXzFJBM1Ys0;N~TeoJR{N^v#EUqbwn(S>b#63?#tzFD@9`qDEV+~_zS@a^Q&QWUZqTK$ohHIM#YZ#YXzqsB#8`3U@ry}l5YZv3JUp$kcTfC5pTJam!
EL0oV2HP|AitjRX3TYHo9GfcVsbQTW?VpuTUu(^xzjce-VVJcF=@jyZI!mfyYi$%4%YUnOb((v`R4>(>{#Vm=Ev}z)=!srqX%Kz6?$)fxA+N}bKDhag)n;?;
zu^aXQ*oI4<CNDvu9|;o=m{6-dnF<+Jr7!O?cY?>hT+ateq@Yi9J3WM&N<dRm%*#Wac*>0W+Y=DX<Y4xHrA=6BQ1c*<&jTGpRZOF^~w3kh!bRszO<J4c}biB
#a)TgSgV?R&)=Qr9_buxi2J%q3ZfQO-)C5{kKzo~pBmccZ7Ynkr+L$!IHMrqKiRY2q(|F!=&Uxk7PinDmezJqM$LauR@B0aoG+gXqwO=2&h=#LRa>lPy}Oh8
&D1AYav(Q2+j)U%e_LQzM*S~QcA(DP-#XW7_5LyLxr{rAXZ^Qx<0HJL%5_d~T&f&rp6i!)yGQP7=6;52m(?6U)O4;h-u_)R_#v(>KbRl2%wm5Z)iH+_Mm<Qn
$j98fax%1${;8;o6{;rAJ+rK@)p3rjd~&1LtETItDfc}n&yY`%rt@oI)M5{Jt!->sJcIILos&7%x$1kpWZ$KAs7|BW&T9F>lBn%8<@!%c57caw3o8be5A9MI
Jqq$oy^6!GN+W#5Ypye_J-5^`*U|YAW6K9<-?XqUCR?TbpdKRCOKQ5-tGbESTs74#T=Qi~tn)EkGnH%hrXv2<K5gVaj3ew*wI83d{x3h(%zbU@V{2yHEjudP
p*^X-4%Po&ajj4FFvS+~*E(PON+W0Q;COmNPV^QkCRK0B3k4BVTxxq$k<p)_-iR{y9caGCIG?Iv?OW$udCPaI(dI_ndB6J)r6+Xbes5=o|CW83#=YKy9nh*U
*6eQHdlD);cUK)uwbmXm0p75^`YZRjRrZo`o{*->wF3)7&r<E^Mp)t+uxcVpxbEd-)a`nfMqOC@V1jh$!l;j=^;O+tQdT_UAoca~em>(~4O7)al8pVN+S~y7
(p2>FNq5!yDOXb-)!#MGO}5*AcC4ZpM169Dz28T>=Ak}|y7v3(DUiPYZYuV}^N!W_yS{R?`#RLKAs<oU9$~HDR|OHPDgILb$X(V)&u~0(i+#L$55_b8Ec?Bx
?q5|OQnA-sahLXaXU83i1>|34`||m0HTMZ>t}e+A`yzd1hU@4}4mt<@tgjqxeMSAj8?8aqvEBO(vPwfI*O}i_7CEkZBxS$VU!t?}V?p!+so&=~_uGtc{MNrT
eDiqwvE`1VRfm4i@o67EKc_fi2Ibc3sa8El_4wzkKdKM?p2E=Qq@_I%+pGZ=JFZgP@^(?wD6}W{@ZJdbD}P%MF}ULGwnfo1G|q8qh2zK%N+J$Zy=RN#`#p}w
6+6#!&(G(!Gtw-i;j1@JH7oTE$ll1es2;e!<iPyNZmQOFX)1JL*=qG3jBvls{$zarCGHp9<NoY){i}-oYa?r0H^5fzyT4Ls(L=3`46q-Nebf5s-hZf$f&2Q4
&(i&TezWVHs&zi@b2SPU+dd8}j+*4A;?QvZRT8!L6I{3Zsx;yVo!y6AcOT+${tQ>hw!%2LUjCcUzf&6hsSg!JE}@<#+0B;Lb}zP8BF*(N$2lwEhQb)LJB))1
T}yh2aW^Kz20h|uP`;)fy1};1+D}d054)-;e3Z_&dY(S9PwATzJqJ^*HE4d`fxe7WP!zv&P_M@(uK7s&!{7Oi@&kPgLtXbDX-%@oYqi=v?@Kd!NYcLh7weol
i+fA{-!rG4_wi*>3vHhp&!steRoXJ&V-T;OkQcU5KK2*JsmJw)`L`B#XMC^U@}rhrpU=!H4((E!`NOVX$v<7m^Oxs@Ms}9<FzFCY3qmKn$NE-B?$OWMxW>7}
y8bY(uapyKwxhN6riu}YL;rd{L;sc!l+H87d$M<SXgcS5Kfdq!drhwIQWScG^r1aP@jbOa$HP&KbsRk8J)w0vnXykxhUWU5V=(!NbM%{*vY11S{geB5RDV_+
v&wc!zx7al)x|bzj`RO&uA#ne`yszS+}EnVTXneSZP(<ZpS4d=znFTw*W`v4t3F?8`?5)w+g?i7f6-dJ`sUXAH!IQuq-FkY8`$3QtNL5scFeln+PL<iYT(i@
Pju`neX^VTRT>vZ+)?bF4f&0}*`b9fR_h71J?~!>hCi0pAzxmNYa3b*l8@H!LDtw$e`H@RpDsTxJxtoAda$K$DrVOAXqchTDn@ExeRhmB+VtM*<vx^;Am!Sf
>UOVKztjFY-ny&irFX%B-_j@^Zs{I^d)ynVJ%6feCDOVZXXvw=Jb&tEQ7_4S@9XP)J~U4oQxWfvhM7F)Kb+6~({bzzt_dlwP|l}b+$OH`mbu4WbxY~w^;~D`
?fR8!RI2mt%!=MD^?+)A7F)xvV{QL%>y^?{hr4f1&)3fR&wsL`Uio=OzEID3&?VNU<@?9-{XcQsd$s!;W+!89v=-^UA<c<=s`7Kisfropw^a*T<{EsH!sySH
ZQbFxNPe<`duyc)s3s!Yc&K9(J&SrS8pw{jmw2Rgu-WeG`miMQ8STxj_JRBJBfgZ@bi4QHSN1n)&8fS!Icd|oOCnYsn~WS-cHlL~i_!qrT6daYzoBOv;~oO(
*6OwD<Qnf2c?WEo_HqTEU0)b6vG#y^e|65~bLzX^qFzh&hpCR$&iUjn_e?3~Qt!U<6z%;i@7emcmGXlVY%8@d`>Nhl9(7Oo3;Fn0oWJZViP%E@DF5w0D~$bc
iF*mZD2Y6_)SBBMpT9-!g_!D|2i0v)b)S-YpU(8&SFU?kdGrttb3Q2l`KM&$JF^(8CS!e(6ZTs+Lorw}pSec9-|HzGah&r9>7uO|yF)7KuIg=HR}{6xQ=Mb0
FJCnt+0w^&{nC<H8~Le!ySGF3qt!t+<SlFdn{7W%cl|v5+q-n{#x~o<bl*nH412N9Hdel_zI)Ks8`jusleQh2EA<Ip>3ZdC`?e?CYp6cvqe~C;dfe}RxpW_)
VuzQqLi<(>QpGx_;=Ch#ex-FT^BMjgelG~Ea+c#f#cG57d%5-Y2YX$|IL5uqdmVF1s~qCr4NP(mwrV+=2mNN?`l5)hTiaKs?SyIx(t_@`Hlo<H*s<=h)>h?z
9(MoVmkj&0dp2}N&v(xDcgL0LAz0=<g0cQRScSFV2Vht#w2?9H9h9w9y;3zoX&mZP*7eGBCfd(vec!NOkUzK+lFoH{J07p>y31tm3)$4Y?)^}mTEFr5(7&V5
Zy0vCCtdbW{{POx_zi(_NA=LEA6|7d<!+O0vpabocCimj_j|0i9no)Onz_!hs3`J5-Di~R2C_YhA*;J?l|F|R?i*F?-^RU$%iK@)_x$ib`mM)I=bPvG_XEn&
(mnaoORKwf{F-a4`i+}x!pe-ESlNUHJ}bw&N2Xgw{3jo({Xf}zUVA0I*VW6^%{?fp^(ps0R=Rq5{1#}XZO?n|$=vMTob>OrCc5`!fqh<e`@OyS2YP)qAO8nX
O9KQH000080000X0HGPezcmN|02q1z01N;C07o%jc42IFWiD=TdF`EhY!pQpz(FB2keH%X2_h6?EJV;W@CS%VEeR-zpouAj$fl{=0*M@twm@S&)0(Cf5TPJc
C_)bkEfuhblvk?ZAiN@pH4zizBUQj?f=NNd;v@0s-fpuqH#hUme6zb%FPr|dyYt$4eDlrBH`C3t^Jcv?uYbRl{mQcni%QJmtfw=xo>?$CYkX$bf~Cc!#RW_9
mlhWm-F-fzU~x%N{CUZuf@MYV`>_)zPstuXZhU4qQ}#D_`?kBk{W-chwC5LPa>V)N@#}!Yyq@1#>N&~sj6=_T|9T^kN2upxKa8|j-)s@&Nn!URdwz$wT=Z1r
<Eq=ceJ3x!^|IZ$pPRn<iJFUB0?X+BNmsvFxH`dA4mUkEGkzVWIJ_}8er;#4h4UWjp8dXG&##rsjH%HFrba&k(`xS*6XA3aYz6XE&WwkR($A+K6Q(6kYBe*$
eD9Wb>hV=P-+#=&n$iRd`xs0A<LN7vbXVHP^S7#x*4LoqvwD0>=a*-9{hgN+o^#*r3)^M^X-N9h9WIuC11VfXd-}gSwf6X&i3`Btv;4WA^YX#q<e!0DPDViL
DKtLA<9bithcxGl0n7(OF<ja9sjogHZ%>BD_0T|WH-=C0Uzhylhbtyey~-B@>3a=M{e*@alI)KR$$aOxqvZXT!A);4^c&fFZ%Ft5dfOGhe60JGiTkj&_KaJe
>~cz74k69;#Q^GGy6R<bKFG;Ye)~?XpR)b!OJ6j&a$f~e9)@H*hEIB?5s-0|!I8Iypk6c!D9^o;tFq%OZ26*|Ptf*)m431MVF1flwO)lh(nopYmJnBNlHcdP
4|%M|7(VH(ymC|@U!A$$nKtI&DrBxE)~je*>prKMWUUt#jvg}a!nCc@wAWwUJb})&)!kRVycTxIThHOG|3Yc44__8uUxo1#V*MJPKlb`HNh?1(9kSfnB+JuU
XFDUkK8H<SJ0K~yUaya=_6b{GZ9!h&rAekw*E+ma@AfLPP1;}T`g4i--n}l%Vw8Gr$<2LeWUc>`<k_Q$+O0sT+TI_^zwu!HjkcUezsQYW?{j!i@jaZW^Zm4Q
e`{bkH@~A!l1;$pb)LmX_80E*f0sinca7k1<?KUfT6Q2!$UNrU534SSU(JF%4s_3^sNSL4zGd^>lEXWIJb>&@-k+^crj2@|D>i7CZ_Mnj9MxTEbAw0R)o0`3
^|Mg>F<E;>9=DbcFF4Uu>(H$yoptU>J<qMT#FWQPqI$Y*f2S&slaF8aFsUkEvE;tHd}vzV*Pw~5!tP&u<fkK*v)2WL^#KXD-nZ{|+rRp!$6$|Zc>H77jt#Lp
?@6**+JEwSlgl~j19IcnMlRiQu9vdo=c@E+X``zeq=~iPBKbJH4SBpy`-knL=p{>^CHr%$|Exq}Ep;FNGjwbDcE)@E=uQ{wJLvRU`Sz9~Hz9=+JX&(3?0+U@
4e)sxl6z5qHPYdb`mQB!);Q$pe<ZQzu~@{KPk<6<UYJc@zv}n6oJZC;D)l-N-EU8tc=vaP(nomXCT~g48~rt5+@!+WMS@%bbg$~qZ?5>4$Jh72BmBe2tWV1-
eE;sCq$8;1_R2ehe*Sf?nc&NUgu2I~iJTrYrzUURx4+yQ()4{MgF(wzXu5-vPQU7@2@c<`XUXduG<*{|ou<e6rh0E`F3%EX`FWd<W*=)8vh@>Qf5pl}-*=^s
-wcf0OWnVj+RNLUax#VG@5UWJzV_So3bc9%(<}V$@5pqS$nw%e3fKSkM3$ef@|2S!rf>BJw_h=(+m~K>1MkOgE}wa&YeJdF<w*K3QPi)z){jjje}2nd`FbU4
->TQ{hNh~0H8vL>>-r$Ob_p8SuetT?wd>XB;il%BPP{gg+ZDx-_I1$Vq_&f|AZ=fPH6Ko%b7s~5-4u_8zK&n#nD1F!HYye#@Ovb|T8|h}bcdf_ug<B|W511M
|BYu~ZrKM!Jg4YB1#<lh+Mi(AlY3)d-o*Hh^A4f)*z0{%y4Q!XDSFPU*uElNj4Hd6Yd0Mf_1H7W@Bi`IzR}`Q>bYq9R<#S{k(Yi|J3n1=FZDUZ?ZorL!si&{
*f+@PgZTZ9+rBTtu=^9IWlz|`2seTjk0^QeN7(HdYTqSFQMrLlmfTK$9>7KP{rEO<^u*q9{Mm=Rm-;T=zB@JhchU~N87VwA<@VEw?xSnd7CuQ`=Ub#c*FjVs
c&E>@_A!uDJs-o0Zwp1cpJKgNrP#h?EH-M72RfdU)lZV!pUTgV*z-H~I4SwOkG}aTum3NFrC)Ch49^WS-_AbQtSx*|dvM<QMq)e+Ce}GWvip=|scYUv?|kS9
?s%O%4+M5T2-bW-?{m@dRL99!#+?J}lsV#s+RskL8qva&W4)yNdY|H_3ELk@k7r2vBKQA~`11>5dKUlx+Cbew*Nt@1o+pIL5&It6|C0HuDu?x)7sqDM)}`f0
pI`RpSFaXz&!}>UHGjeze?jL(QlBGsiE-{lE5o1a|21o8)b9@EajE+}!8YdkA}so|UPPyZr7!Nh|LVBhotuE%E$NS@s9tNAvwaSMO_uv*l|$xt-m_Ygb&CFF
!>L1xr;hU+9&bh@ZTi@05y^DW?t}h+2-UbZ(NFCZrrfxV9`A{s_aI7J&gq#iZ3eq%>5r`RC!fFX1g-~1!RyRw^Prc(<%G2!hI@XAJY~<>lhr5n+Mlqg_nbbB
;Wj)urS0*>Oxp9Y?)ilE1X<75bKHxEXSWag{vS|F0|XQR000O8001EX`@`BArW*hN4D$d03IG5AOHW^NZ+9+kaCx2GbJG~evIg+jwr$(V?j{@CwryJ*+qP}n
w(VqNZ+xF~tLxrAeR}Yosrvmf{R#T1>YkZUrgG_W)dL6W5~yd&MvXf*?2s}~qLlfXXG@tbQOYK5J9O$$zfIk?9U3+MSG`F6RvjCo-mzKzc8zgQl{tI%ELqc~
PM4^6qW|)Q2^uI+pkNRjLXt2L4#Go3hye*mB1jG?NJ^3lQbT&k3|Sx>WQQD(7xF^^QV5DeX($6_NoA-4b)g>Aw`e1@hYrvYIzbObFX#;eVW`D0VG@}P(_p${
CYcM1g(a}UV5PxoSW7mLjSib&2kbJ~1N-3+9EPKC0?xvDase(HT!t%f9d5u)atrRkedYmq2#?4Uc*eYdSIld8L*6kTgpcqEzA1iq2o(4~1!jUkPzXjsE5brJ
2=5`HK_rOGL{UU_h^C0n#DJKJIEr`{@eL9{LW9IYQYM)qxkXBYR3wc<T89jtGLfu8PK#WSTak~+&lG^)JQQ>&1cgZvC@K^;DB+<bDNV{6l=D#DpaN9%REbnJ
s3KGos*@T})1tOe$DuA$U(tYR2#q~8g=VBVv|w5)S_!QcZ9TMO+A|$UN2Zgavqd+9?iM`^dNREXdXv7;-(i5oKxQ!fB@D3`3d0>nC`K`(VT>?V80Rp7nFNy!
rodE~W-y)1aF|JEk=Yh=gt;&e<|`IBEEE=z#bk-aQf3*fFjy(9CF@|lu)$%Y#U|Knutl+zY!kLKI}LWh9<o>1C+t@ousFyZ5)P9iaFiTlj(a#sPFb85&M;@;
99%HCNG_2p%vDd<;JU#LxM^{V+%~vlahJJga33BrkH};4M0jTKH}hP0&Ac&qt9Yk)Z}EYAB%c&tg>S-lgCFoy@yj4kkpC3eB8Wv$gWy64MM#TKOlU<I64oG`
LwF&AK}1iH3?h>#5S58WqLWx2VheE`;xh3_d?A5FLM9POOp=hKkP=ccsYx0|T1dyFx5yx56f%*_kVTQzAe$mP$sy!qav9`S<h97>sQ~;Y{0;?~Lhz@euuz04
$`o@bE|hR6X;2DE3uT0|q#RW6P?4#msBBQhQ&kVugc?jui&~y)8`LqVYf;~!fkh*sF=@gyHE5=2F0}B}(xH{mT4-a?mT5=YGaU>%3Y|$8gKp4W(S!74dO<%A
{XGpZ7{m;QzZ`}zLz!U?!##~K7|D!+(PWHbti?Es@fH(2O(c^HCc_lPRACyKZZHF8l38Rn%oXMvEP#b%5m~HQ;$f+<OtBnRC{{YGg4JXVtQFQV>zR!Pn;bS<
Y%$oz>=1UsF4znEnFGQ>IHWkNI6{uWapnY^Von>J@o<(oM=mfIh0DSfa@E6i;f8QiaSLuM?kMgG_k{b*1K|<#n0dlHgTLXq#S6tti#Nhs;hn{M_&`3A&+rAl
l5Ymz4Srb!3i|JY3c*Nl2qA=oP!JlzFyVyoLIftFLu5};6j7OIis&Q;6H^fj;*hwC_)G#t5=BxE$%Paor6QF<Y9@`47Sb7Hgv=hY2-!$>A*UjjB9B2{$Vc*%
0tUY+e)sSPQ&9MmDN2e9C76;>N+|76#-SXPhl)%kMO8&Lhw4z1sbx?b>L}_O)HA3L4M;<w5os(m5t@=_isnKK($b+7(;C_cZH0CY?G+uMBk3e`CS4R=p&QfP
Lr>@>^k(`%f5kv%kYX_T3x<%P%y2S-j1)#WjA6!EjAteZlgU(rX^QF03^LPT7R*-6fw{~)GT&hlEH+pIOPS@cLa|b@N?5H}L)J4J6`PpNWQ$@e*+#aL9k83(
!|Wsb;ef?Ki$e~Fg(JdI;h1ooIU$^6PASd^XPI+~^W=hXQMlykip5p9W^mo$2Hb+%<PO{=_n7<e03IqHk;jTB<SFw^_*-~RUNA4oYj^{1$vfsf^Fi?mJ`4ZA
7vURxCqEQF;TH{3237<oA(+ric!;2gWDuD|AyFNoK}-@0VnZB*cqD;ELPa7Wu_B2<QXv_WoTOk<ku*#OgNz25NfwK&Og16AMGi$ygIo@|g*=c~kx$6a6!7qe
P!I|Ue|jj)6d^^S7*mRrhH^}KQUNMiR2HgORE27a>WUg3YC<i8+7@+5eQ2O)2#rV+p{dYJXaOySR-`qwVcHtBGiWb#5IP%lfvyJKNe`x{(95DX^mXV5{b3+8
$YHQB#9)}g2w@~M8pgm_GL9L~OjJxJ)0pXG2AK)7$ZW+NX0F9N#eBs=VUe&{v4mL)%V7msX|c*+wWl?(mRU#E3majRht13u4_k$8upM?Pc9Gr89yq`pbU18r
#KTd=F>>7Dq;SgMv~Y$wYjBR7XD%o%I$UBdGgrt}gKGvigq!3R+!gMV2L_MGW5pBl6kd3EDZGN$@W$dT^Bz7pd~*0qzLIaAz8n02Uk?Af%tQhUK}j$OPC`IP
CX^7`Aq*29B0^*)ia|6I9b!OCMJ$8Z5QoGyh$qDNl)xe(BvK@XB#NYpWF&<}DoD+wVbWTp^N?Q1pvb7m1eq0C46-_8v&b&wAUO?kL2e-r$qV@)KT|;Q8}qyH
hebh)LJof_3Nu9vijrbN2~tu~T2aQJEK?pTSX3mHNfm>tLN$x(qz0);YB9A9>X5n?^`Jh}K+y;qGfe}anMHF&3uwu-R<t4QnD)?t=?I;mGwDjYk?ukd=motk
`jEcRkLfQA5C$p+!C*23hBCv1;mk-FMMlFI7|V=<@eUJ(Nfwh~3Nw|NCQSD*Lot)gVrC0-4CXTP$U<1;X|b?GSZc6LvD{$=SqZC!HL#XhCu~q`WHt#~JZx2L
haIq!?1DXty|9lQP#hEv861WqilgMX;)KOXgHys8<}7nQ04@j@g-gt3#Wlrsa)Y_);TCh-;*M|^?g{sq2jn61$l<Z#iQ=inGxE2^b9mwKlDraL3vb{pykp)g
J_sM-6MP|G72n8r#SiAE!LNU1m=OOiFcX9X6@oD#n2<~;2+f31geBpH2o4byQ5>R@=nk<gVmriP;tKH~J|tigL1IY4B!y&5awa8{3R06aLRv*SA%jIml1Y)7
WHrbpWG6WwC*%_Hki0@ZlAjb1ekXq@3MvW-g-H=o3`!_UDoR0VgEFM7Lj{M5OeIAXQVpszH54@!wS?MCT~ZI~3k?h!35}tNLo<sOLQA17(~fBm9Sk}tI+N}O
JuG@My`VSrvFNMl2Llv?U@#fN41?h?QZWif8;ntmwHQanlL?B6FbO6LQ^-_d8kw$`!ORq9!E9lUVlJ5n^O*$}3l)nLOJFH1gXOS7u~JwCt6`17TC&byJ+r}K
6SEn%2wP#B#dg?1b~3wQH`$}uE9^5kARM$f1c%8{gJX*07AF)Z6{pB)IK!M{&I=cWiw>6zF2hxF&Eh(_VQ>>}8QfOfA$Q?E^T6SuheyI=izo0Do;f`C@PfQ#
UXj<r8^v3Ar+8032%m(1;0yB=zFGV*_$mA%|M!@hB8VcGBDfF&LXt2-SO`xdF;PfV5={}E#2_&ZVp+r{aTIY$JVktqgcgZ}#E^s}g=9hslF}j-NiC!i(lY5t
ddNUBGMOAQ3t1epGTDV37CD(*BsY_X$t&bj<QEE(Ld>6}Fcg7e2E`R63`z>6gwjG8QcfsuQ2{Cnl?|#GRCTDv)O4ujP@B|Y>XG^e4GfwXG$YNS1++A1rD#ps
khY%M3muq_OlRm~(N*XsbXW9%o)*2JH`9mog?>VR7ytudkYX_W6+lB_n89#|5zI)7QOsyz3>hnogYhr{CK^m4lND27Dl<(nU6^4olgu)hBg|#y!2*MYibaaW
o|Y(<DwY{6w^-q@QnAWlHM53UOV$bN9X42OBAbOR7F!kDU^}w|cEWD5SJ=nwCkKRsibHVN;)rmR98;VyI7v=1rxj-uXURD@Z*T!F8eCFbg==sfZdlwTx5*ug
yKoQgD;~f@i^t3pcq%*-{x*0Cui&-8Tk;M*ID9ntL_Qn*17F}P`6hga9~M6i0)_fdK_I9>Fhy`7gb-2)C4`1B9>O|=W5O#UKtv(3rzk>HPtk<v7BQHZLM$OR
6PLt;_>jONp+O>xBo;{($w&&4l1b%|8q$!oB)vrjAv2Se$!3vV$U$;KZXu5%pFw`%H~8J+4@E&mAySwWCB>NHObJrbpcGSDC?k|*$|=f21*W1wB}HYZ;;E{l
nnQJ=22;zTwnZIM*P))GzM`Q+Ba6nKnp!kdG>2A98%0}1dxMTl7w9T<6S_O}fL@B;ias9tk^T+?6ax*0SPUh@gyCca8407vXp1q5u^z_51Ysgf3V_KLQxsDb
GZeEdW-I1c%q8<Y%x4xb3q33{SZuKjmK&_FSZ%R}SqJMC8x$K2Hj&L9ws_b|wkftVJ7A~5E{oj;`<VR(2NVY#4#N@VsKYVEap8n;G5}6_I1OinvmVaD1rHY;
E_u4_aK+PAa?Rkna3cV2dAJRC;I5~89`2I|@Q`@~kC~_BSpfWP@Z91hyfSzVZ!F$2?>xO{K6?1%@R|9?!xzOjgYUu*@{<G#9Vk#>haf^w5)6V{gzym3A(SFC
69&RUI3_$30U|;qgQ!AuA%-F*iN(Z*I2LgOARd#zA)z9XLt=|0kTd|2Tcof^X^>ixR*_DTUXg)hWHNimYLFdr2sueEMQ+HW$SdUYkUs!^R}_Ro@Fyu^P!x)J
D6S}BQA$x-DC1C0C=V4$C807?)uI}y4mAvFLM@Bh9_k2n74=Acg9Zi-Nh7AQK@-yKztCLK!k`tj7TPfFpuM7lr;dtFo;oYKc<9db5PEv(4SfQrFX^Y~&kP^~
nL%W*hat>RGR(tp4<ngT!WbB5FrJwpOtP4wn5LNFVJ4Zy%r=-~FxSI8VLn*^3k?=Aixo=@mclY-IkVDWmBAXaPO;u$qr)b$*~1pt3fqM34m%V(g<T$Y!yb!$
!hYd^#X*BZio<Y(9ED?^j$52?IB9Uo(`kz{a2C$N1rHZJT(-CZR}HRtx-Q&M++=Q9+%~vla96m;+&6fjc*Hywo(NBwX94iH;yLqzd1>*=;I)T0inkW;;Df<O
@=5q&@zuk(0QfHaQ2aFbW%0isGz)<th$5&#FpJ<!2t`OmC?PZx#vrUiI1-+T=n$ESLZS)Lg&0gsMJy7V#DTbqc%I@b5-<rZ5-XB8BvmAHNDiqyq-N4US|*)`
^duu>GRSO@1+qdmhwO@+Bp2ki$iw6ffP5rB{N|yMLt%%aq?n?(LkUkM4N5D@Sd?`rPbw%XT2xY0fvO&=2S81z#ncw+Skx8j8Ptac77ZO5Sv0n2;?UHgSpYN_
S~#@y)QV|s(MHi$(axbgbg<~?(242Hbg}3P-8^*<fF43mrkBtg`Y?T=pGAL913V0bLBe3gUy31$p~5g>xMBn|S}_L3!Z=~PFo8_8n8ZvLrU+AE8kue}!_!QI
Sq8IV4w(z{$pTntu*hMt!xD$3und-aSYfb|ta4b*tT9;YVV%KxvcX}aVw14dV4JWVb~x<xu#4<w_9*rW`#c<QIOuQ)4#N?0)Wb1yTsUEIk~syZ4bH$>i*o^V
-opiQQMjbI?BNQz%3M=ix3~ef;5OVbxTm<Uc;N7eJQ1D>&n*5nc<$kahnMh5cun5GTj8DJJ@WxRT70tjZ19h#FBV_no5gqGhlihvUj~7~{=2{qK@>rSU>3nW
gfIw6LJ6TA!dQfba1h=i0z`C(1d$D*I7DNjLkx&XVmZV%h)d#;_znpy5(<fhBnC;DWCqD0C6mfSY9@_EIwrl4!6KtWCPii;i$zvNHbr)lL&zBbxtY9*{00Tc
Z=QZP_(M?;3JHH&6!ugEib63(aZ*AkX;4a0+Mrwjl((p$s0fuDs#sJrsP3tTq86!bQO}{iqJcppi^il0H1*I-XbvrymJV$#+9}!_bnws-Ix(FUT|9I(=%(l{
^t9+@(c7YrMPH%6!vKdt4uhG${tH83sKGFY;bertNP|)TrO}EpFjg36FrJxcFv(zw#Z<*Kiy4ZUidh!36?0&&!#soeWP!s%W|3mCumqM`EOS_HvBF@b!79aS
vPQAiV4Y&UumLtQn-rTZwgkXd58DlP!Y<g&>~YxZVV}i*azHp}aflpdj#wO395Xo1oFFGHPQzJ?^KgN=sJLWtS#d>imAR(4F5ED<Np6wb7I!S}8r);<GY`lk
#bd=2Pfr!k9G=4qgO|)JgV*qedF${_c<<o@d=x&xXN!L<z6f988+?Z!4nN5+Ay7E8zy?7*1S26mgi?eS!dQeA!Yd*;M1;r|Q5>QQ(IC1*42Y?SMdDh-W8y;s
gM<c&AaMXBF-XcJ^OT%PK~h30i!_iH(mA9TGB{*ZWP;2j3&|#Aha3($J>)XTZIFlLg?tA24GIL%9~K223K<kJC<?_4iaV4LN-9bVWjvKtl((qhsiIKHpo*fZ
MKurA4QhC(X;I6fwud@GU8rYJKL8p+BZtOJ6KHDC%%HhL3x}4_%AmE-hG|RML3@J^LPrmsptC_2i>^#JraSb2o*sHx^d@}_`jY+@0}Tcl3}*gv7-BKhV3@@S
gOM-_Mmvl#7%Pmk7_XRMFwtN#GsVMHPt!ciATvG8Qp{G&v6yQykD2dbfyF{tBrH}eAxjm@JS{g^@n2eHv0AYP)_PdyutBj=v59Q9*kZBOV!Oo-i=DzQgWVqX
SnO5oBl{Hx6bIpu!4bt#i{lO_3{Enq6sO4<hqDIfJe>D*fn2n>WN}$>#o?-_YZliXZYXYgy5(@&;7$PCRopYUFFY`KXz<9xV~Z!Ao|0$GbMnICWdOYL@Y=&0
hqngr6z|Cg4<9W)8GMF+JbY1nCEv(*haVO{J^gYB9NsR75Y!+T1ZP4hLK=iJ2<<6M0EA`2k?;l)6cIf{GKd@iQ6MTrGl*^xLlMhEY>4A2t|FdAd?ta1gdP$>
Vv@u|QbjTk$%Pc2QYuoB)Fh2XT17gG^d2%OGJ42lkQuT-HiPVn9G-GQE+IFQ*Hb==0v5kJ6jT(l_|u`VMG=FdOfie%1|>X{RFtwP&6Kq$rzr1G5h^oP9I8S!
i|R}bQj@90)OM(&sLRwds2@NL3>q3V5*iCl9GV$4SG2HbY0-vht7xZa@2P{((V>%v&WbM3mGm&^>CoGtuh37?-_roaKxU9I7={Q#VVJ`R7%hx*81FDanCM}$
#gqV;N~U?3?rDa>EQ2`~a~<Y8EO1yD0E-+Jdst$y6qY$G4}cYlmBQ)(T0_<<)-f9lHd<_AHd}1*u$63Mw!;p^PQ@+{yA^v3_L6-T`<a6Vhb#_T9D$?c7#t@j
gp&rR3{G2|5zaE_$a%N`7d>54Tz0skxT?5Dt_wGqo6Ic_w-t94_n7;h9xx9L9(j7~@WkLLJQM!5cy93mUXoYBYl}AqZ{fY-qv8{McKFBPi>I%OZwBAthrv&a
Uy48x>;fAEF$fC59D-YffRIdRMHod`gYYDRA`*$rL?KZjnj$*Hw20*)wh+f5o<)3z1SFwFB8$XA5{INhGDUJGg(9VgR31{3G)!6#=@jV|87wjxWH!hGS%qvu
c0~>kISq1o$Ze6wAg@I}lAjb1euF<O3OW=L{$vU>MLZN`iaV5GN;;HsDD6<jqHF+^Gbj%gnM$OxP(@LVR2OOpwV*cCVd{FQ2lbf-LPLi}4vjrDF=(o2X3<>H
!l9+2l|gHvjnEd_Idt&U(V~+<7Z2S$bSFK8p3qC^P5OA~YtYX_e=>j>s2D^B8~o)k)M7Y{Fc=A=$Y{kFhp{jY#w#W=lZ464RG7w0XJ*1Im`xVILSZqp%woA>
g<>VFa#+o*Vb;P%W|P5YW{a>Dwkfug9g3Zb-4=VuUS^-ee&zsk5Dpt0Q5+@53{H4B38xgNg)@q?7Uw-&Fu2HE5-tl@46Z7!Ib1imA>3qcDQ+9wQQQ^o3HKEb
6b}s^DIN<?gs1RK_*?P9;ibVVi`N!!gtzcc@c}+5J}dr#F9u(QZ}1&{Fh31`{VVTB{C|N-P!fy@E`)Fh385j3A}kZ$Lqt!Jn8*;-BDxThiA7>V9E*5D0wy6O
wn$=;lq7@X1}Q933aK1YGie;sLOMlygA5KCNhZh)**s)d<gmyExh?WQJ|@4X0>W<=e<%tGf080llqm)!p|nLAC=2DGf>4R6MrtrMp%zmY>O(`Jk*CH6O@(F#
%@r-6mC%~BW!eiJp%Zjwx{$6+ccF)eo<c80AB(;L(9c7EivcjmV6eeZGTdRLFp3#X#u$u+ab&zO!P6vW3NzJU8kxb&gjtF?FqfGJ^A!u2g=7)4m{~%W!ZK!s
Vx?jgtcEon)+*Ku8!R>%Y=$k&R$)8rfStl#vX9vh2jCz%1c#X;!cmK3!f}Ta<Rm%GoH01-a1PEJT!JgiHFBHWhX=w#;gP{(<|#ab7vv?pg4c>Sinqc$<~{kK
_-OG7KEoHqSLPdhSNw!u|LRpE{inb}5C{grA*4kpCM**U!Yd*O5fzajvO!dcricMCAvP1oATEi=#P^gC5<y~;M3Iz9CL|{*NJ=4<A~ll+(vozLfn-!<VloTa
AP19^$)(86<T1!=kdG;#_+3#@Q3wh{F{U__5=skYJe6h2k@6N5pt7P0scKQppt?g%sKwM~>X3R+-=Kk_AvA)<7EKJA88i=o7EDWv)=V4H&O>|XV9`<N%yfZn
Om{^Op{GMH(nryk^ke#a7+^7w42Hiv4OI+-;V{Brq%fKpD~wZ&cbH@_#bPQ<bC~Wh17^W&n8VEVG+$W2EDnGr%u-LwnB}m-Vx_@qvfg5Y!$#QTX)|mw*lMxO
VY|Z)*a^D~cEcXnXRsd*7#tK1SsW%u6i0<)%n45?6{ncfaK_=R#W}@!hYQR_gG+GP(-nuS0dx&+33r6M2KV5hr^n=p!!vj;ydbaOHN0WoGVeURCm$R>!Y6~z
2LC9&c=#%OW4=55AU~O3|H?j*b%8B{cnaze41x<GJcP6e#e`<UFkvAa2~Q$IB#6vJh3F8|B9=pJ62~B}MLdJ}OadVhB(_NDkcvqwq=QTbSs<H1c13QB{6Ybt
AQbXcl9YndP?4#`RJN#UP{X1oQy&_UCW@xe+Mx|;>#3bX`vB<Rp`)Ucr!Guarkl{+q6gE{p_kAb`Y8I6eoTL1fMTGBK?Z{j{!$EKhLT~5;RYkfNHPjWE5?wq
igC<%n7~XFCc$KwqL?a7Q_K)%!fcqs%!fr}F)V>)umV=X8rTTCVK4a~P)h>@6aWAK2mk;8ApkQ$VA!lI000d0000UA002u*Ut@1<E^csnoqhdNTgR4dM?%j8
Teo5f<70ymBRk0)1)Bm(U_1naA%Ks94K8~!;RcHjqr6x3>dmWrU(GL5^H<D2Jg8pp@?vT27B#6-#YpF)d-wiad-v}DeD~?_==X2JzlDGQ@ypfUFaP%AtA!so
{#WP6#f2XS|N6K8_qX%^)Bo4MeYqNbzjyw>|9*A-{onuR`TxDT{`()7yWN!+i}~Whe=Phr|C_3Y5bV>u{v5;K!%yb?8~@In-|e#t{|<@k`CY@$<n?DJ{=0Sk
S-k!nUVkpHKc(wW<@$5?`g0_n^X>KLFV~+3@fnX`o;H>{*PlB4-i^<nhWonl`b9C`xj3#teC`oE`$2r>2l%c|<LSolhhx1*zg6Hplkxu{epiowKaOKqh~u`g
O~wB|jMp`39r%-9e<tGpho9Q@XY2a2ef>GO{`9UtefZrz^Vgpu{@ds}6US7E-__$<HRES1-qVZM+uROezRh^e$@S+7XlHz2<8EAkR^l@c;F)#c8(X6%z&}&)
uHDG@Z5Uey&pC`^tP>w*ao*)fhvC0huRnY78Sh~HmADqR4oC5w)4(^!z>mZ4XMsPAMzgTqi;?e2^!(vGm##lAu0QK>9`kW*J3!a67_)u1B<5oC+@!T>0>2yY
ejmqf&nx1!4WM(0c)S^}AM)WdG2RJ5=aa~*9T=aDeUt8Q!EdjwKfhjoev7|d#OK)fhR<6A8Q2FIn2+}`kJW%5@&0|_SLC4q=@FDw<f|bqmSIlAcb@@0SLj^_
an5~^?IK>+03Pp0y`yxJ(fT>ja}Vfgylga>kF*<J{}yOFeAa#7D}K&Z)P1aTKg4^6wE8LDyByc<tyn+PHxmvIGe1>;-p6o%jr1(rpF8l~A;>3QJCwnf*PnBe
_Y%Fgi0|pd>oT%SNR!8)6AnQR3b=20?I(~U=KrBi_zd)78=+)~^_j_A0qgxD@^BG#tLas<ZDtFf!u*iu9>;k#;<JsH%}!L}do!U!=V`r7Mo^zCxxWDUo1wXE
Tz_5zt^O42g8l;PuncUH&0#|Dqw#(LbYB+oaYoPI1%5R;7h)|d^mi@NX$5G!4Z2_j=v5_s%W|JbdKkYifP6H-hscFKwKX+bKZt9IyfOp)(goeR6rWX%v^9B}
0y$=#cmVJ0(>fO*H^a60DEJB23hB89dIHD%DUKKCI|aN^p>;&NgnIKd@*eXY>gg`b#qKwLT?U>veQETXpmk;6N7?xtbaNSa^(6j=w6^(p^I&^3MS2+d9ewZ|
=xNt9fX~?W+M2(N_I3&A@>bXn)LjjtNfGHc1#yI}p^dcxvSBepNcg0SU?-}Pu1ZhjAiIjM(H3D$WBREKHh71gjj_t6<G;4!bw>~*E&?AJ%}w7~JY%x)G+vM6
XPN}!ubjqnmEfx?JZAv+KLbDTB*u;C4;65KE7EEJHhm7_L5|s~<OiICpEpDPI_fFp?G<6`Ob@a@^a<8?TF6kFbivSueG&3h6LuEup7A`=w;TU$HUZbfc%dBq
Mb;baQy_o+L3U&=&JSaEq`C5o9z}o4e8&6qz5?`D2iDx|meIgsS<jEhIG_RgrU!GxxC3R(=z9TtXY}rqepGAQfpw{X?|4djYf<<jEyw4|VXg(mR=9?z@w!#e
l_t9;9}U0(%#T6;#eC1V@p+$N&NCDfVcc4$_$rIEF#4}i?2o>H`G_Uq3z^>jN!U-0D+Yik8AkJc{H_GHprBaYY!34FfaaWo-c0%4CqUy~99KE&x94Dsk><g{
z;e>BMa0lDz#{fwJp<_Szc~KcQsm_e2UkG-hd9WD<G<TH{s0{^5ov3B9c{V!(3_xJYG5PL$727g4bOTp#t%cR@CVRKY}*&f2J!loVeU&5n_dxJc7!j0*V{W$
|J&MQe0Au^LK))ng2t8u7Q_5l1Ak{l`1ojN3RtsMxW7&DNDt)WJl=m0$Fm1A#`L}fduuk_bV@D$za42~?|R_)fjPuL7~k-EBCnp&x-HQ)J7g2tH^+F?{E&jM
MV5~%z#V2`zK3w%b7u@?@ZFv{vUe80dn07QuD<|U8I4h=zJl-U@7GbseinG+CSj2i4t9x{5XU$;&J~K&X24FqjNg|?FCBw_Fb8;ZCSn-#%O)xA&d5GwfRolK
k1-H=kC%?UVm}hEc>>p<4P<#SSt*h4&;<L1Hu(u)%@2Tn6%h-~01qNfIqvIHyz*@#H-d43;SA^S{=0++O%g2^fzG>PZJLlrvV7I7z_dRAEo}?~z{9FUk13dU
7kFfj?jQd5jY9*ouN7LOZzzAkJZ!$%d7OIy{{1%2^&Y(M5%|xX+hRJX*d!xtJCAe2`Pf`$Mb0V%nxL#sQH}=ZjQO<^%-z-@B`Yiw1A*bTq8wfdjF){=g%$6B
YzM$O9)P@;V9XdVwg5Z0C;YBJ>$C_l=z}q_D#}I>`BD2F{cT*MONU479jrs<KyJs${12oX>hO*|p!q_4-<*isXXyJNe6tK>Ld6Uf!k;+T8=N)4c})vBW}R+h
_>0JWcz7uDvHf-!=f=J^zweF{7d!=inFab~kk2)Jt8#u_kb@T3j3C}&a$lh}xFdASAz>ItV53#ud^ghX6Y(j^(GlbnkiTsl)1b3a=2`Ak4T146>J8QpFM+nZ
Fux|PSqJj=949OLjdn02-Yq#iu_)H^Fi{_Yx^oWvm<;5oL^!uSU+J5S<O;FCm!Q{5grC*OC;OJhinbH$LYN;Jj}P(Lrkgq-lj!TUz_)0^I+W;nGw}Ts<s5On
jZRa<FSuR<;=>uLRmh;G!g7YpFUR0RJOCNH8~<*?9O{I>o834UdJ}CG%kTwVZ*!~(elJnJ6!8ModsVX21ER@Xv>Eu`;MhddRZ}9)?}Bcq0<PA)h5uS|<mixi
F%#Ia`5sDU>yT$Y2iiX%Kd$NAXSw7BvI7|B)M2erk5@rwW#HFTiB1{CMFIRy<)hpg^IM&8b<`R5zW0&l7!w7m-(kMYoM)YruHZVGW8p`#ZNR_Z$G>MG_IL<u
G)sP;$5)TZR(}I_6X~`Pf1ggoUDLvTXB5L){DQd-)}<|ov5;OS-)P4&@F9*!CmxBoGC(ZBb^`fzj^wEhbmh9aBY4gO$lFel|H!s+O6203R9BdhT}1tevR@-Q
Vx7dHgV}#idUzp$^I8oS+Fg4W=Ng{U`UfWm^U3k=(KkGHd{?jj(eCYtnvp5;V_H<Fin4|_8ROn>M4ZdHS+4OJh&mROA(T(91ySq4anKIvmr0Q?d{6v26LtD7
`37?|4~$2%Tj&zhBR2O<@XeV99N(xszN!hH6gsj>n1$J;D)|SQh|RFp(erDM$)~s9j|CopJgesR6#M{tUroefUY*7a#oVR%{D<(|4A#6(cnI=<t@U?tt*0Qy
Fy5JTXpHd)^1R9m97CL2i(_E9N13ra#X0E^#?T&;pL`_p74<~DYc<k&k}w#=T#iA1*jlG*`Uawo$=1{O|2OcXr$oHPG(;IJ!TRu;A`YqYWF^9^j7Jtgk7aaS
m2|h+#R1G~hT`!Him@J2O*iH+XQ=)>6EPIWP8IMSyTX1e8`e$a#K-BQrw;B~0$&?3uBNaDbHo#e@ts}B33__<2hfQb?GMNVW{bLAeV0*fN|k(-J?D4iYs5@a
_zz$6fG{r4jv&vUJ2`@5QMXrxcs~eC@K9iQ1MsI8z~8zfaG~!hUpte)nlYBjAnxZ{g+my_L*bWHof?)d)f}O}Qla?Y@-LS#zAEv)(P`GfPmMkq@LkQpnI2N@
h_~K9IK0BwW#kv9YT;f2P45UBbsqn&09_UCbqDaf3SBcramXCda~gPfhI~sUGbk^HDPsO11O1J1j`(Fae%B?x0P{`gcOHQ*<r?d0;iG!^G}qe?<>McPUcE=p
#OLC8cFDg-erH@_O2j;<M?S&5j5&6loCwP3J;#ph#%mPL&b(G9z8fTTw&!~W!QVR&CkM(^Vl8M8I)6ZWW|XW>f}Q`4{Iq+3@9`Xt;eD5+e+CIG_>g?C1(A>8
^>_lYB-*we@Lj7krz+u5tWOO4vAI`-Ucj-KKdb!Sj?f49M2_Z&WHf+&QaAxV;{(Y1UBSz!H)|BXXMp9+3YwPy$E=EaW5nt+!g<lhtBTqH%pI|=pCz4Gr+NUy
5x*67B9N}C5oTrfb&BXyq8KtGJf;cq#eNam)4JnhV7{>eYl3@uI>N5u--hjFgjJUkb?<@j1Dtaee5WJuHKysFN1z)SqbUWwNAx({+`~Epyf!C%x!d9kT%(Qk
GE=~ZnaHzF3;bpV;@@MkDXDnikZ{WZVf(wJ?;cT(#N=4TA!tt*$>*yIpQkGL@iEAZ<!}~Ytf;3RI(~IE;p5$by8Pj|{u%VuOR7hjA^Vk4%>whz4B?+1CQ=m`
IPSqyevgW47Rf$&b@~~tGv*I>;kn-le8B3T_5f2vpN4HI;sYxFQG8%?;r$S1tEL@n7h^kx9U$$HX1gNy$}zu}&tRRl2=NK?L<wlcc<)`uM$ZsN!Lg0EcmFB)
6<p6&r@YoIV96Em)w+)T4&<YC$%d*mwRhLR??Rtpj_TxC2RDhAj-7oZSSPbceg=-m>K)LApuV%UMIDflFLaUc1C0JyvvVo@N95y0iYYvd?j`Ay+88|Op3ony
=p&vB-~5tv!<>_6&jb!v1zp|(|Edl;XA1b-?ngae5_#%LhZbC`!uvg%B2H+6p37ie_`XTtqjRts?3-||SLH{z_5z>9H7gadXQN5Et`g{L+=o^dx-^Jf`XTYO
7bhB@RU$vBdWI^k2aW-8FN?<(9c&KopQX47?Jw4wRiP#ddE`pem>8|<fO!olzf%Rh(1ClZVYKhcIn+v`te}60zqiP*UlhE4Byxrpn@@^bC$urC`X|-XMy)&I
FMk1jaqRF?i((Aq{W-|5+TU2~h5BvI!2;@JZ)XT698g@-5_v@I1yd1us!Y&<bGaC6Rt28bgt+IPz^%;go|CQ#RHKo?{!1c`t~hx<)EPm<aTQUYT@|*8Yav(%
FM`g_oV<<NtB=0bx8y@sAXm+{@sR3rQJ-V(8*zjt$UX8-pxj!E@RZx&J>S85Lb$pA-C|DUtsg<Y?$U3MBJQZ-LyPN8uOCs&t?;1tPMnMT1STDS4Rvn~^7`L{
pC5?7886t3@85MWg_m&u9VgetSdfkNfyfJD9{LIB?_=Rh&53*i_7xh4+P9~`huANQ=VgA?)h+9wCqZT(k`Kal2kZy!fUi;~%*vj@{$EAtGpr@wB|S7P?)U2O
uvVJ)3C+;niVNpCxaL2?o(8m&8P&sl4?cMl@bVJrxLu0jzJalI$=^j?c30SE&N(5zgY|xxGhKw7j;ba8W=x&&wBw)M1^u}I*zX5XGf{^+`<hdSGz)%}?FBj(
`z+8`wfEPcmV65C#ad!Dk4eJaS#B7&x=T1hMtY0sxG3s!jAzj=M;~p9>TZINC$`H|AVV``r|RM69x$duop@%)$q!x%KW~csyiyqLfzzZs|E^QxyBTTOC7*nO
>?Z0@)x+#a;NMT+o`I-uWVw1A^d32-OQ*=U98hj>QrOnu)OE03Yk^EODc7W8@JUev$u(&zr(-laBp>&o<J;FooX#~vhY*X*5DxN?Vp*Oq*Jg8{MXouVfw8df
+=SX6mYZW|55u%mOEP@-i4)^nj~bKLn#j{$kpCD&odo7-?@~=Jj(OLaOXgq#AE3szN_lnj1s?!i2Vt~d+A(3emttQC&h3L!%lg#Gm)9vDrRt72r}kdZk>9KK
L8fxn=vU|9<62Lx;r;&tSi>aMjXVJw{Ewjfg<gufXzU*{OL>Qi*her+`Aps?%K28L<6S4O`T_VY2n<@)?;VP~z<C(m?-K~u!`$;Z$#hl3|1W7wkE!POHg&P+
yQ^3`mE&&)y`P+8=8DLZa^Kr&%CA<b4s%i1pn4eHleZgmf0aViLuDej#&K>IM*FQ=oo{e_1FqM`c_Ll77FN}tlmOGhx@^V~#_ciQ4SwIJrl^_W{YF))733Od
ulCUEmGnqp(Z_@Z)(E%293t1Xv;fazy^FpE+vSRrGi6@j_4)^=j^`2RYV@r$k*m8+Y<lSI0lAZqZ>*nvI0hHmr8<Tp=$;MJfTv9ne#Cpwa6i`~;fe~!RsJ7i
&vnoP*mIj}#4#57ByuQ9U!PNssR_CyKpcrZn+KxqdWzzht#GqHiR$B%g|9uohW%!&zsPVM*626u8-#s<%0c(Tv3@G{EbFsmJxq4Q_0NLt%VN(s3;)0W&9Fz9
^&_+~TF=JM;bw0r>tBOC%CNr<_Iku?)@iO6L=PJm_I}IZ8P;cM_^yxPYfqa&812bx^cmM{&H6VnU9f+(*Dq`Mz4d$j9B%fkN%fsVnlL{Y-6q0lpEvB6Xgr3!
)3H~%%?*1bt%uQm>#4p~Ss3m4XTM|LTJA+;^f?bVdyehF{C|pLvcA~(eCwHow6Xj0aI=pr_gh_}d-`FtPt*jBnfv7;-{2Tj&q!QnuNRW_h2lQ$n_;x?G4==V
k<8=u#;^Co&E8s9Kp%VFHt^Pd{LbXqWEp#eF(1@GH<-L&Z!VN6>>1hyx^I(As$NdR_rH$sM;^5Ibm%?UKeZ>;+I}0-1N%;5U#L}~H*xLvNk2Rne7*%VQRC&_
z*|H^)e}vPi~G=S0Ub=2tkXO3+1$^r75rYcHAhbs@b2ySJ@-RKTe43)(*_we{gH>!zL++b4TtYgpWAiDV+SCcEIX(h(0;0(qR)fwi?4d0o8G{F$E=HcF!uHM
eLv_PY$ktK@fp^Ok$ZisUiS^LX80ZUW8P2jpXxJuO4mLY`tB?~+xQjrj_T!yy1Ef=_Kx%B$!8aEU%7F1pL>LRz8sDP&njS=uh5*?w%b?^$Zn+i;9mkS=3~}8
ch>QF(0%>bKf!gr1U@pE!}+`jH+%H2k-cG`kZIlr-xSyGEcoYJ^ynR^Uk|{RDPP3svkf$w6f(y&H5;``_SWkqiakVCPeIhLIM0os&jDdyr$>Bm`ssw$rx*XW
zRMT!-`vyiXBrdxT&627Kt~#l-iNRKQ&E2@Kk&?v3$}}_4_HSPz~^S$PJy1O9>eG(UI?F1>C8O7n|qI|{;-=M*OR~}-1~Be;)l1SXY85Gi{?Xq6n<Im=u?&x
<gJt~Rlc0@zB;>r`>diLutGK*;|}YUx(YPl=TA`V@Q!>p)00M%7Vz_g(0^>RP4~ZoXM6#jw-52L(NgKHoMZ~us10jSyM-MrkI~=g8?1=?O&5L)U;9TQFKh_E
AY~)jkL!)`L248i?2BvWqb?eER#GRvcOQK628{*dM)ceGoo~X|ezFswukgA;U@{o*Vz1fPaelwU`@e|T(ay3!eQ3Hx`PY3%&Q8f!Hrud7d~7-;7qJ?y^~M<c
oD)u@;u_Q4tWWXpj)<j;Tl#s}o7?LRt^ADw=CKq;d;4<iiD&T)XiaJq$Ku~>0!!)vZiRBZ9PinT_w2#-C%_YU#(~L%_42;}*;*lN0?!~*XINoBa^n~5f1L|m
spPty&|NPgUi}%KhkeCWeB7hCm=C!xFc!vaej>X*pt0=>e_Qp<K85Gx!f)>gyl`Idtit`6ju~NfMvn`?AeFwvb46HR)PybQ!}A%tRsGfN-e-;v*9MHu;-wQ2
C#?eSUWJ?Ir7S0Ss}V-~@UM#4oUz@G6Z^2;Y>9Du{<Wu%{-AfMxuAb=Li%}~Xi!kBr!bE`tx+NHyIkOho^G!>m>Oe6NW1qS!|IHn6z$P>L%q9Bwsipd#ft+`
$D*A!eTP2Wz`-f^A$~x6z5)3TwoM*ZT!Z^N0!zQ3G4dL1fPcw$oO{@#FL(v-U60pIkRHkfpXFj64T_^}&HoBAZ)g4B+PB1*|0(D``RL2p`qco7MtRA^bFHU;
nbw5$qtWLL<cL(vZF4~T%yPUhWQFx8=Zw%Eah~KS$5;E6VwTLoGMM-HoT@yGo>e>H#19PzW9fsPN8G*v<7VFO#_xZjxJShmJs8hG#7FzW-zpORWs$IBWwNI`
6hGlP5J;DmpwEcxlkMZNY=fP70qbt(#+*61t~c?%=kYr`OY|f8Zj9^gk&k!^dVzhSoMOJ*$w4B2zKP?XAm6JX9%#Y!=np6x{4S2O9q)TSCYQcOz9r_O7%xyV
vIh3C9rSsPEaTgdD`*gArp|9dzT1a;3&I<{GnaDGhYF8jKFk4ood=xql|#pxlgq{Y`Fz6XGW;}mY#{Pfhx`=A&zA)@dL?A<fa)i9L@mU!s2OOHUAJ}n82K6X
2J3asSrn9CXZ|^$ydUd4_K&6kzr`5u4EDFYfmrWPu(P(7^CA!Zj{KdiFnTT(+v8fo4yfE8VuIe8q8A~3j${h|>N$D$+uXk)a&3mMa@>zH@C(SaI_rn!=n|gs
TF6a7H5&ZB7W{h))*?l-mRP54!cyDhXP#1hhQ0Geya&JEiahr=`YT@|9eMm6h@DPBchnr4vM%y_oWDj~Lt*f1gflgqx|u%3AkQ4X{mjXs@Htxr_`a_rz1kwr
-4^u={M{F@k*^}HE&eIMN4QV+5_9DCOssh-h&Q%D-`c#-C^yJ8Cufvn<k+DQafI2$b+Wx2%g*1zkNv}`t2v?g!s6mSU~3)92^ERBtsK&4%}<2U^EDZRWM0C0
EQ>kE<<GA`x3ZjL?bm$=UpfJLye6Jkm}yRR427s;;PuNXmzK)GKchTrL&P7-r(oNgCoqIP;B`Lh&0?p|lm}cRdh!_1U$Yt-)N^(|JJ&OLbvAv$cM3PkMXuhf
EBJ$IEm4<x^$i8Bb4Ofj*X@IhDW9?B#H_DHe9twQ-Z{@)Z?;D@Et^6nQ*{@Je_=ew>y7fD&but4%rTC+2jgY=H=3Bu{8R8S%Uw>qY_#RG2DvT?X~_Jn&NEyN
=`%Ikj-P5aZJzLZg)^G`DvV*C?m=94N#xAE^C3}2yz@=-1ZM5!H~TOyFYe)5We?xvTF({o8?a7^=Z1cxI#YAcsbN@!XXlCh!;YwDe&)y&#$X$yZ_Ll<yhBUm
(F}XLNa%!1(1l#{i2QyLM$bjv3!`U0m&w+#58>ep_lZvpmv~EgfL{nZx*%K`eO!!@-jLl=xL+>xEWd-}(-o+h=lJ8VBA348;DyY89{>CX^u##t*_iq=WxM7R
v156Rd`$=pLe=sx7K-_u3872oiAJ^-ydJ7<&#Td&Alj}9`(}1!hh(rrwPdL}7CsO6HRKIeDXzJoTw!0--EL9dbTas7_+CLQl8gM4;Uk5p>8R1Znge)_s>Nz2
@^#~K*H=NG9sC(E7S;`^yq0H&*k8qb(5a|%!Z=m!FX8i%=LvIDx$bhp@BTy7$M>C@)xz-|@wZf66UY8-!f2RY<+#^M;l=})m)S42rOJQIC|7g^v4>ZqiZ})H
8QK>QgM0^kr0Caoa+_t~-4h3&D}>B4hQ=`x`shl|52zLl^Dvo%%k&fVzWaceww?F&0cY(C`}cFyV~E53PC2!eL>(IY*C&)`W;~Je;Hla+)+1fYf4&y`qzu#K
ct0oIhxTQM-n&BaT$|RUPJXw-=T*Hu=0-g1_dsAs%8%W2^x_WqeGTwmdQk7?;c|>ovHe>nTo3nRaov~km(q<n)tWMf^ak|!CB>}D-f(Qc<K#LJPgHU039QjN
Tstp#)!zTysl_(^iGKJz=r4QMgm`{xe^7_U)gt@l;pydD>}T<Q--@3JC(d7W_VuJ{-A;s$w?jNph?+!iUqT`3XbZvs_eK28bzF>37U0t$p5tMln=lsSJ;t%H
X29mPCiJ@6+thdVg0+P|V0_<m>MPi%@%9f`Y|#c@P<(VL?EbdM&)pYsOip7cV5}TpdU=fb#J+(Y@X>kyNFjWWO;JmXdUI36O9kmw#-!R~Y8W?32GAFu5c^*_
Mqd*+bulKdj@Zw#s9(v0-jiN-@-m+(rsg`6w&Mf$M1IYl$9ix>;KDi8d3MI^nKj?yv#7Z$5<Xmye3@RNmL_vzwLal>&&b9VPK{w9@^G7g#e29T?&X=FeE5|!
XQpuux^oS9>Z90;u_E@v<icj4&i_dDHJJB6pWo=*BfRrI<s?^WpQw!oF$UB}_WVB}hR?^;H(e$6ICUHz=Q<{&O(FE)0oY;Ay>d=f?MvKD;5gfkkIuCK1?k9V
0{=x_)f2eovZI4A=ij(R9qQ_s+8fhFzdHN3xUN^pN}1|E)E;E+zhOBo!#XgBRV4OO@H~}W^Xi4KLVAyX0s7c}v%l5#7s97I8-o?zX0KHi^gjJsB7QBP7H}DA
OjNDjs^f>G#?=qHmx<aBb|P{|KPPy3++J_)vC?obw!+!7*$e5t%_oi?=2~}-ozPb~5b<k4n2n<UmDr=8YIMxL{~qlx>RQxmmqG{c0shkfyKK1;+rzaUu`Zn7
N3mWuk4;hUnbTaC$!_h58h6a!esT6~SPiz~dzG{GFxQl>@@$u<FTMzzVTbkwPb6|%xPNV3V4tX0)IN#=)=}kAQnj6Jfp4o?k39JM(7oDmuDfG<#{9HLHK}|L
?*CV@je4fC3q7%qQQ_N+1I#D#45@uVHBm!^a=j64-!jA<nA`mm=seCJVBFWnQH}MTOw*q!e|Qdl#Xzj<FR*__<;Ylvr}j-^t?3Tsg3u@7y~60*uZ5faxm0aO
BT--doN$N}8pD!9=M<)bKKPj<AHR;t&mb1}6V;l1rrP(_aI=>L=0CO_3?2PB#-q8e4|Oy8(QLn}!fu~H?Qq$#8I3XV<5C#icfBU!JKm3flE_D`2Y+8X*QX#)
7=0#1z0NzxgQ=RJzXINfGLwosRBo^%F#IoLbgQDl2KYCM2RuJ;Tjb7OP>uofI$qt9&FeYU!l&d7<E%H%Tx~p7_tXJfy%lcmmtS&lUr#1+FC^-c(@2X8k^|g>
^mBaHTiV~SMA&~p^<}GL_B43w(+|2w(59#bZv?;3#<)Cuk81n3DL08aY5?n#lFNejSFjIvCF=1DSObrDxvvZFH+doQz7vFxZc@D~`#hhWF{t=l^;6ghy4R1%
7|TBTqJJjVDJPrN5cNT*pSLKsSrKt&+1W#by3=$&$AWJhY!z!$mLZm23!}YSRNg^ho(E#TjKWN~Pf{xPa}ji3pFQ~gnW+0Z3%Vz$iZ4FWK25IoOwkc@itF$l
-*5M}MLl7vURL?V7zbWK9!&Kvc}uzT3&&TgjoH7{7doI!xeD$RcPVsfhx`%F+i~vqMDX89V!zsNVSFEe6Q{l$^*#G9YM(puRWIm%v-==ps(#+<ABwi{An0BP
xyX%qep?@W=I6o}Psw3pjE*l0%oo>lnRG`v@vgH(ZU2DQXG`P<i=g|}q94HDe-Sw5ju=;sVq)%%x-W7$UT-`cJC64^kWt1<8ONKTd~}}35f)CJ3+Jc`!YNik
&;CYuE$SCVOXhX2-g_^M?_Ib<KK>cS_OBskD?72O!g<<+Wj}M|@GOk>Qa&KwQ?*gKz;;vnE>|7=!u0eK)uCLF-CrKFPq|Illeb5!xaEGx4&@U1fE5&pT8wf6
H@ix#{g#t!TyktfYCX9QbU-?ma}E1K59d_dyA#s=qz{D8`#gbXDZjQCM*EeQ1umuVSWiFPrjGV~80~lL)&HT7aGTm(FAn2c7W0qB?MoZipHAhXdrmDPV&)f4
z5XjlKB|dY_W_M#UF^H)Iqxz2w3etVC`1l!HQ~$F#Qls1AV&Iva!$WG*f91<*>h@_CIwFngumtC$Elv%WoJ(%^It*s8+GK4*xRrq;^emDL*qLB4z(R<JGLpW
)d{10yOEbyguUT74EL|_`jiuUpmSKekMM8KpD;eKEo!>f#>9j@sEI~7L@X|cJ$Ic1y*I=-m?xCHcs&L>P`k55HLF~Mb^!Nx!p&akswP?ar^=VegYL6uvV7Y<
iW7<Y#gB=c9q&oKO4ujfXEX1_UD$7HA`z!yjziTt9Dv`%K2aXh{n2^dCn#5XN_n+as$FBctMgZs&9m5;v6^2TKTFjasM_k2s8ia)w^99d5pPiE17Hkd>-!e`
Y}Ef7P^aPTPhF*Fs{U6vSFYbGL~SbL3Ti*xQW))7G!T1{Ujc4x&zYwh46McAeII$yXOtioiaHJRs)O+LECkGT8IQb5;CS<4v|s4e7;FslSlrY0Ens0y(pOus
{yE5SkMf-=XOI(quoQWodyr#pgnhJH@cY-YZ;W+!dzAa!6!p7az0xb-N3Xt_^Loa|Z=5^}>ywM1`v`40z7OXDtPkOqh%vnWg{Ptx=^&*0LMy*Y?Y(B&oeJ#0
_6j5JdIo+?jciLUY=x>fdG6$skjFohJ!}gssqOf2eadxv@x0M03!}a2RuWiRs(uRbU6a+{L5Hx9@mbV2sMv<{DtO%+_`V?hmJ1A|A4d1G7r{Skt08KE+2^Sx
)<xxBdSSGWp;vdepV-5(P0v@p)hWa)UN8QiNf%rQ-+y21$J}!G(#t)y#XYKa5oxn8YWnlwpJjqRUY5uoo`_r#`~J1WUP!M;6ZRukyo@xqa{>A6)=z{H6r!e8
?Qgfa4gcda%MYEL6Y~h3lWt?Ovyk!ZLpwi{&v@jsmA&&KLl`|v*v=TmbF%F$N<0J6_#lMQ^H9gtm{sGNhoH~BwlgXlqE;DormdOLtWNZ^b1J`sx!OGO-Kq0)
RK8X1H7^`m7)|%W=vk9?O)cJQ*Wy~^S*Z3spMi?=M;^xWDv`ez!q@Xn@!VYGr5e2d3FNA`!p;7r<&f@y+$LVJb2W?L_f}yWwGQ<9$;q2yzBqSkgQvo1uZe!R
c|L55bfVdGWfw7@F-{ld!bdv=9a4yVz!JroleCU_#_FAL^Bm@Jd%|qpxHhUrxiaJ>UKiGdoWo;Ts)>C@9kLhqX-pxEp3TkJ4&pY3kz$TY<#;e(Pyt=B8}vC5
<LXZ*oEp}NL>=w$KXopAmwcXn9G|j}O^VMf2gkbTkaT(#^!j2P7uv;Yv@b1@%jEN1dr%)ZfOR#QwJ~%-58HRhZ)$JcVYs>1dX@0q6PT}^t8BWEZJx5*3O8-j
{+6<+xkUb~gB`*%&r6Yp%|spjfb8rbjPBd_Y8kH*bFkR03A{E5V{L`6=dVr#-FK0(#_|{}4r{r*eb(<{o=EvzXvd6aO~={V)`UOv6#Rq?@@eC09{8O5X7Kxx
?^A4}eE220zby9ksTw-8KZ8X5!<CSMOQ4yZU2gN+CqHK^d_AwbPW}M;5mk^=^f%F`we!MDpgXHT(<XduozLTVTLRns&8fLi{X^OR^7`Vku4w^(pnO%q$F{jl
!n(}CePyb(o}>8Q-uE)DGt$}4pI-o43`Fds{PJ=b-Mf88IZ&@(0oGcfuYfW%8~wa)7(JIA<AP?md6qo-rzP>*5zw?YrdB9K-kFK<R~u}?64eW1J-qF~GGCVC
uG`d}_knMZ;=e5p!M$C4PInQ$o=LwOzV5YG{fST(OR&x@;Dr+S><__?R3Qd1n%|56&M3A(dtz&dzV{BqcqQ;d6i=jLMV!w|p;wG=@P7RL3FsQr5B7|gppUvR
{soHh0@$-E(V-w-tPuSazZP-*1JOc_-)tzxS(W(g3dP}7&;jSL))mk@EK^7u^I<-Lf5y6Zj{e5E9tnL|C;e|a$;NOT|D8oz9L0Nvzdep~SpZxhL|=9$%2^lu
p>7;^1)kp|y^+ya?+{-a4Iakpm42TD9&HL;_EPvBk6{kze_R4Ro3vgQpBn~a>&rf1ldu_%|64#il&t}bvo3yX3ST~hbw^u<*JXq=G=ayuK=*rbE$h+WVx3Tm
&nbbuWM3{&>@gdEGrBHDYzFB*2hU}>m=eB5Gu}4^d^1V9s!BSbN-+xh1H5;gb7Lz4b6%l6UhOct=lg=N2gE+SGk1Ck9CwTEZxCkN6MoWd_pDeBp+h+{b(Xf_
x60SISpTP>&lUL+MtcQueSo)jpX<wTFRsN;x#KTig`4MN;+ZVCHeBC#75p>bz4NNM)-ZL}?7YZbv%i9U7B;AUYCh=x)u)ip%?QK7{Q0wR^IQORKKX{oX>E)A
2%lxI_J7pk_n3FPfZTBbV>%HyE!Rw3g>;|ERBbCi?<3{;@jW@!sjNDCLW>wjBX+=aP_Z=kspDGw+L)SRRcF}<H+%Z1-g5UvU937+F?aIFseH&u80{;CbW*(}
*PXpuHizOC`B2=yxdyo_FGu`G@XxXR82p}ecK?0av$G>;$at~W+nmpAWSTFFSkH?GIbKYiZJ$#vopa@R&}W?D{$~#(DpS3U*Ov_Q{Hjh2b4#YXxF`I$J$+@7
<3n2DzKLfJUya*e@14`?)nMm<YbtF01Z<Smu;k9OP7^WrHtiu$^|~r|w-s*osy}mL%{t}v-bY_cVd1KV(0o!4i#UXQI>!LUpG{$xN)T6L+&fKpVi)p}O%ZF%
!Q2LrA5-;?+_(0WFnXMqilLvx-*!YSz6iecp|F4KSDD{y?;XHcaBfE9RK1qhi-q^b^0OBNro{VPy!|RCBJZ{&e6==wA0T$d7_0@)GM{}BezQ4M0mH94Jd0;n
sJ^OqpcaCAGpKzGs5ew^p3i_!d?5eCdLJWwcAPyBUd>7yYG1fUBm;TLXf1Q7Ut@otW7WMeHL{3(91>R21gy=g$-}zo4(RkM*|ieM`JM2!R|{e<6^iXG_OFS3
8s@76kkb?LMRBa3<FmYdk!IVbfaYzdUhZSKd8Vv)HqYP|d9$a0-5_?1wD|$_CD#}Xgsn>LSHNB%dk{yW%|addSm2EI+wi|{V9bx=^<Aoez?u`}F<a*b>@lz!
$SK0UXF;zW3fk748k~xQrPqK~+u`fksclggP=kDB1!`K*hd)cy<II5GJC4s_`+@r}s-iXp>!3NWi)*qNzV`c6{i9Gn41|C7&cV`h(lrl3CJh%ZD2B1`Z4Unh
P)h>@6aWAK2mk;8AprF^u~>*2005Zf0RRjD002u*Uv^<^b!9GYaCyaDJ&P^Nb!}yZqY5hMpo7C<xc3QOA_;*p0+$kJHck{M5)bJK1;#S(87IQOLVr|0K_6xU
H>z15d)2u&8RpJRpYE=z{jv7iYya`ze(|sW{>vX6|8xA|^KZWW{@34q{-?+1pZ)e9KY#uB{M+Au_ut=r^Pj)^{dd3l_OJi{FW>yf_uu~I|G)p;H~;nRU%vn9
M?d}PkAL#@55Io=@5B7}569pB{pp|o@9Q!D|M%_o{p#!e?~eKL@2!50KJPH^r}}k}*E^^7`9r)<c7599-uE>>zRvsoy)Ay;=imSRV|wd=_EEoGcZmCGOm08k
zkkf^zG~)saraev9n#Oo{#`3RN1qux(r@Rcljnndzm5EP^?g*ovitHd@3$?U`ziW~u~;W|e+*!6`mF8y>bX0n<}cdR{K@LR+QLqv`$X@H-7oC<a?imK_xEFF
*0q6sxq2_}=K{qc@Be?zzwa-Q-|XVPImr9zll%K-etw(U{yHYN-_UjM&o@lJ;=T_xe(2)!tlooqp4s<!k6+b%<9_`%Ie*{9{c{07bpPBwYjzFAtKD25HE-*t
A4G9$HGVn7^+Ws9wR7lxko|SzfNFlt?x#ksTbrDnM$f_eqt!WH*|R=1Kd`co$K>LQx9ROeHjZkume0ETgY`>l9E<+GlYaDm-9f(JLE;DOdppc?v%*DKdzH!u
kzYWcH)hSgx%(EyTPgoFX6?DgVn3d<_S~!ZLCpgPc`ng?q5Or7|5J7h#l0-9)EDpTe*II-&r^6Co9~{B=NRC<sQb~iPY9n`-Otn97j0pG-MEzfoF=XpiaQ6H
ce!|_Yrk2)rN(I~oUWU=8v8yQz%NeH->>#l19oXwy>B<}%zkC<H@i;P<l+l-KHYfn2%R?-A3%RU%ynbq{oVVj71xiw?qgHDewg-YSG`fK_ub<o_xcEbRreY4
cdlP}6LE9&IV%2ol72(|c{;yD>*U4(t9^j#C2StR-pAYg*q*9+Wf%8zx{g)N*HU<fd;e8_0$ZPP{Vn>vRepJyaqBg+`#x2_-NnOc;5?#y64?v2u2$utvADVW
{?vTct^2fD+aIjoV)+Z|zE$ggr?@Xxal$<=zkq$}AWrC{Ut)0rH;!=Ep@DPE+V7Lx_ZPW-V|wQi?I(7>yYVo}lTaLYKNsxzXj8-CSv&9M328nhTi0Ui!U)e-
KWAl6S==Or6ISbsX&mM>*U81f&Z)(BEKgGP*AFw!MfRkNcw4Ft<nkF%9gXEPq~cxoIo){v{rmU4_ERxl+l6yh@!(WH4Qqc~d{yN+y*dwL?HtP=J4kzh_{zKL
OJDk{%NIraSoQC(s^7VOUd1`xxD4?>QtLd#^>*<p^xhC3Jmuf2=fUN-xaT3wi>U7Vs{d-F-#Uu%h8xeMJ}>1r*?NTASBdy!>3!<<W2WlBtGJ1c?{~#@RXgRz
=jz|H`3<tumx$A_aTps9vwb#cyUX&~+4?$*OCjE~iU*`|*-qLugrBSZI_SPmfKRG*Nw-f*-CtGSR?3gD?*Y}XhWQ-TILNJ!SNEN}pVa(hfb)awY+K~ZY@b~k
uNbqoo~Ztn;w7^EE!92+whvb2pLg+`4e@=a_$#aDIL+&wrhj7l#9HzCYF(>|_lN4|Rh}nnubOCQcj1`3ct|NfSM4uui};Mqn~<G37k2I-zrIZ^Z`j>0<n}Ec
;Jvv0Bdi}(c^g%{A&ui9dw})_s$Z)7D7Q|P;=%O!@p-HEU)`V6^`~lIjT+CTar66rLcFQf`x@kVSm9M}AM&b?$i;QoIMC%GHS^r6aY-6?PwT6x{qCuHxmzzt
`^_|;lU<Ji^1v>}qpTlc^SzVAZy#iSiF5_-*MExf<x$+HsD6O#N}siJtLCA*_ya1hRMq8hc^sYmo{h&)-kRnCrSSf0Ka0ArQ@q90bF+0A6_;`AZ!V7@rBC4E
9giaJm)1R2`*Pj;W$VQ1K2UW$dgxrIbZ=8UjI>=u@y{+^*AVf_6}~hU`PsSfSKYMd?>~R%zkU+)@hWdp#p}>Io~A#k=3~{Lr|N>KxKcg0mx#k6`<K$8-=+6m
)j@Lgau8n4+L>YEZ!5g08kZoP4dF&=ef$Le9o3nu`;yf~PV1SwJl87jj{bh77s#$J(vL!X&s9CCnRp7TPvPP$h!2qFcUJQ@_S~p-8T9)DtYdXQBmMPGKG$v>
66xZNqTPCRT#x$OSlwK8ecW@lYp&2Py~0=Z_o((%-5;!ub(JrH_7CFsviYKlzqt4b((_2^8o2x(R=12j-)tUqj~lPzxmnd=S^mSA+_@U${&Va4?C+oRW85R9
`@~*%75DAZuTSNNDleK{r#BNXY{kzX=6h59W?#f<RX)rC?3&BFdet0-RNu5($7B1MhUr&SJH5)=QvG1B(%WEpy=~KeVs$2V`I#}bb?>HsQR`<aUffK3laAM2
y)n0MZ`T|;w~sevchh>7YW|+$iKpwn)&3*4Psm-ri?m~Ey$bF7Dqm3T<E-k3uHxxc{@6`>sPd%M^Kb&Ygx=d}+TmS&X;pnPwNH%YO{4XEH7@1qi!>3(Na;$W
{+FsBWzQ#jzv=o>^*vu=-Kp9iTa9<CxalzMJ>qSq_|0GH{O{&%qWKJ?xZbEQ_W}0#b86SE&XqxUq*~Wd&+V>0@U))2TJJ&pfi&N!ht5s9evNp)DP9lT-^bQZ
Se@FGj(v6CR{hDYbLGx8+of~n^4zM=o$f<U>zD58qf~K6mlu%QH)HdDNaL&OSH!nNeWb2mM{_JveMhgF|KajZ)cJDk_0QsZ?T)Y1pS!x($PT3Dsg0R=FI8RL
G|r;t4^6c1Y<*hAQL6jn)qTyW`%Ta7VR2RWJ-B_dYClP}?uq)ZtMeq%`ym}SH-IOi^Pl$L==rL1C)v3(>YUiLE)A=TnC6`vV%^Bq6GuE073XGkLaOnuJMVEf
exTONM{(a&bpq5l`Em0PZXfb$KX>vRqW9g4d2q_!SNrqMh5bLLXO~lc3*qjF2e?bW)}0H3@?%ziQsv)Pb<<P!(ycS@n#1-g9ZQ#w+Q|G2)d}3Wyl!0#*$=i3
#lD}FE&!^Bv33RV?9p?ai}%6$A$2|*tJk9HR;oO#Y9Fo3vt#`?n!Drj8Pk0TRo?t+U-$X3pHcBLcJ7Z`pLBUc=d9U#7jH}J7^L)BQ@Sl|-BtB_W7eKmo#)Ex
YpVHfT7S*uGau$TPxm3Rc2$iNS$~!4=Vj**s(wP9Z&=lV-mP1ce($P1P~&c-_vzyA>3FBgA5!N(?CwiX&7r+t@42ZSgXUW8n(u)8Fx&6<F!3!{pV^&DiTv#+
F&{?xqpIJ%dOuePH&J=}PxG9l@kn=`u4~_R{atlVWlI07>aSD!IAgPaNcCmy<{9<keW`ei+E2soA60i}WfxQZv8=sAytCAtbhR(to&U>TcU7mU@-lbTP1t@;
Ru4PHcUSRncRm=xAJFfkI(%xLi>tp?otLfRBJ6%}=cr%B{qUUH{;0lB#0Nh`95tP%u=Vn)9(Q%_(ysU`T^HTOPdvi^&;I-p^S;OY7$;8QS5+NI*MAN&|5N=*
ieIPlfKxbuiYp<zyW#_>{ES_Fw60#C${R?{S#kOQtUtfR{7BWoKs*MPXEY|4&pjp=C#vV=GWQLtf1~+`E^eC22UENSH!i!c_ccGp+f^O*lr9&W@1eMj^>eDu
mKsN-`lHpjVUYF|=|ZRU7|@(=w~sSb4|4SbUH+b`55eM_u8uVuSET2uq;>1m{gC3Vx%~jD?wHG?QRlU>eN1XRa852SaqIG`-CEhr^xQYM-mLC3q_^@mH9Pet
#&>GHNaYo)a{^Sp1Uug`Ro6i{vKl|IdTXh9$SR-JttalDPp;OLkl(sU{G!VDME0SZ?=}7VtUX12YO0=3I-X|dFyC>WmF_q?-$V3U=)K-~9~Wuw-1vF79`7#w
7?u~p*6DW7A$0qZs``Zp_if}pR_D7tMgPU_LwEj2l^0p%zp#DX?71H1`d8~W)xPRo^R`)BY&XAz?H6}>P>9cp=HIw=y{djFi?395EY$sd744L&N7O-_I*RsP
_2cQj{^#WUG24%InsyoOBbV=+o;Q-p8&mUFQ}2VFuUV~+Ud44#^-Ekn62k4Pb=6ef&DQnO@pJX>pT#_VSD#8J*T1Tt?B;<hzr)V0tMZLmeEKl$iOMTL&vAjz
AEJL{=OL?cn~IOS`Dm&i`BnP;tM@rbztAR^XQunJSLdTTFZCqrPgPw36{oNE*QoeWb*@}m*Pyz;cg=@O@kYD(9@&0>HXdT@`iM_|k#?TtVI#dumcPlKt8~A{
tLB4L^&*D(9#I_6_Is;wj~d@Jb6-|<b=^J$w$7T?VNrQCtS%0E?}%@w;)N?5`4ZoIdfs#c??;`di}r1m|GbMgnBt8*LU=aY*OulPvgfeMbF12|mE9Spov!je
*tqI2?HHRMR_6h^xNdb0w~BY~(qU_1zoqh2t8?jg_ur&-_*r|c=4b3&xU}A<o2ROM{VE^IjVE^J32t8!!v9!&&+VsU@d;I*59!2L`y$wLcog?}Tj00Io}9&e
qN*>D(tAqvt+P59RXuptE{~bb`_j5-YW?Y9;-D#BH{y-3d>LeChIswmbL-POxamA3Jx^Dy=X@!=dG-6M?itD>)Vj4>pB;0rzhL*%9Zxz)f4^&vA*-jv>W)<V
#nk6{n0`jhk5L|vo}-g?ewR)sin}_9(^7e(s{2))yQ9Xp={e>uFO02!oFeYJ`uhjCt~(F=zW;u1;!4#%OjIWv#eIePI$hoNm5#rg2eG_QSD&$}uU^G1+4unI
VX*cR&HrHMO;r1zSD(x3kF)xLubR8<_ARLUr;F!**If1c^Sp`g>0!pj)qZ<dPmrzu-PcFO`}@L=rSjWd`fY0e-)>#^ls*8`v32oy<d5Ii{YUfT{Gl4RvHAT;
URRBCSo@6npjQ31NcYO^A4PG~6*{l3-gkAr3yYJpJZcs1K>Vs*^U~G%F9<h5=e9b>U{$Y3>410f`f5K8n)|HcE$%$llwSLTv?sgu!K(Nq;-y#huw495eNI;2
hwaC7d5o{p13<csY@O51^H86|N*_e6YggwnR(W(RzgE?gX7h_9d=6Lr!}s-KaZ%Rquku-yzvJ@d)Hy5&r)BSVg>&@cdtTl5*QDL7&JSbpHMcLF<w;fR!&SZ&
%A1;LSF7=&I$s0LsdVS>y3bv$Uv$#WvwAkH9jf9dDV|Rk-@7_5rW((w^S9CGtNDM5&(Tc#)Bv8pTEAyFo=)>S-1DyLD5&{MYQ9<{?W5}VR`ak?-Ic9JBAx)U
t8V{(DzEQke9H1E-2Ql#7s1YhLh-~9_Yte3lByGQ@tm&g-GjW&RrE(}U+`JnA1;4>*PNfKUuXM}Q}cdy&&k_$J=e~h;`J}{{IWV$u5Kdx_bGniIWy~@o-?50
p{U>0<)5Cwk2Vp{O7YiR{i^%^AK-mlo9!<f7rOd?Z2UDs_K(F^o@V@ai2Iw(@6|j&#Rb*4s)_hvEBY6<o~+I-QSB3|i&u4Bcg=+!=6>AOCxQ5nX+2!GPKK`k
G~>c*f6Zy;^)8P@<+HlH(j)vn*}U1ElibAXxqO3E9&r}!N)@M7@5j~qbot+ipZulzty6m8>YM<zzbB>V;?5aII>{~cp4C23RF6P*q^g(4_NA)$&l%d!uAg`7
R!D!ms!!zZt5l!MF1<?jefE&Q+CA@B%}dfea95w##gVGI=4rihc7L&bOsbyby^llH_im*9VDlfhudK?8RQa^(yit{pz~Ue%Pf`0@QGEIpiKF~X#92@u1LC*c
-|JTk|D56lA-i#s&(GRFRiCGmb{g&DQ#{wIeosn|Mdfu?^FSA;adp#F{p%sFds^2sH75b#VpUyqb&jMuZ{FphtNv5vg}L>eYP{3L^+WtcmN(<tEp?v4-4Au}
m7P1G_Ti}fjg+pCJCB>?`K|tr<z=P#@71|vs*V`ak7$#_3)OkKYCOU6_FSAPHHVPJmC)~3=cV2KiTa+7usy#2{$q0W19iUODeezf2cW78&*G@nxhv{?0=2(=
SAXJtpIqX3aQn~QK0Q|th1FeE^9uHSRQulCI%rkbzmfQdi!)#IV?3!kKat&!Y@F4@cIqhF(cQW;h##!Z??wGjtUkFK_g<63FHf;vQ0<qj_T{j+v&$b=>vagH
cX?W=zK3RBN5!|E<odbwFmygx{@gBo1hwAF@>^V<Str*cHQ%LLAFlFPsydQvKIrNTzOTpI)Z%n^F1;JyKa2ic)yu2)Lk#lVyZJKmuhqC@i1vGRuAeh=?sxOo
R9=xfH!RhEr1o7S9iEHCi`>so^AFT`Q1#DhfATJ!J~S^PRo`;w?xgjU-8qh|zv?8e&z|f1`9l2$tLM5xc2CuHapN?!{?Ezb#3-+2`{Gc3pz7(U^~eG8jdWj~
+b5#p3*9`g>G~zQPc(1H#Su}R=-9N*D|?~hU8=4qyFWS^KOnsdcRsF~_nuRWYY=ZD6_+6XIm<&#%~7uEN2=>`uZLZtpJI7j>E}Sa0+uJE;$ORPqm|vS;-~6-
26gV<A=(}Exo@VuRQ0XWbKl*54p#5RolB$Y^&jE=@9J@(_{yDI=IYp0>l7~TruOk8KboE^$l^0DFVO98VDm7w?;)i((!_N|dX;WGqUNF1^TvL!*16F7js;Fo
tz)wDZ_+y1YQ4$D164j4oA0T*?X2#x%Hvk^Y8P)#%}YS*#P$)ny1?l=DZ&|%t|Z%6qt3y*>#C{virbIj))8GD;az@To%b;o`KPM0oYHS|<HWQM*Zu3KVw|n&
13W|TRlOfJeq`&(?E62-cnjIDYkKd?<#AT|sj3c`JFmvgH?GjRP4VtmbJd3Uo?P92SErk;SGf6^I-fUn|EajTnSY1!ZG@YQ#eH-UcUkFlj-vf|f_;PH&nn)a
*2B~}>?s{pm4C?U2B+psvHc;4CsozStm>(<eb0zz?Z(mf_wY8oeaP0|SzOYsGotHXq@R1W{Z;20t>y@+c<X7}IW@0fbwF61p3}6)tiK%quTSY6x$`M6(Vo4E
$IRwIsvbv*_rUrwcTQpz-%#U@YG0A+f7$*@)PJASt!C>-_uqSp`M5g=6!B6Xvv%&>`b@PCDm9nrGS9h-`>uG*sePK#mmG`xaOa4)`uHb_Q&)8zs_#+7hta-6
{RJm!AKgA)wy&R!W7xi9Hm-H^<rMy%(p_M6sZf7C>!$|DbJ)D+0rIXkx%f-fn`ZfOuD%glpJw0NFzqhucU|4EbUa?w-*xBAvg?G_N7dasM8DEN-jUK1s_I<3
ylb_uI8|Rf#eLHkexsZIBBdw%rF8XD`S&UA8+Sgo%L6&c^>+30s(5@#_uJJ`Y{mD>o*Ol;KEwC5ikrvu?3DWRRewOruDCo^HqN@|1#e{BrRF6pzfheApyEuf
{<#_-sCrWB91#@<VCNH8bs?+zC{-R+E3U7rH?XU({MhuqR9>#Duj|?^SD$@#UaE6xTs*1TN9yW;sW_RdFUHml)wygazv-s^NzDmyzjyhKRo)~U@3Q(VJ#^oz
`Z+pJ?E1TSlY36sdR1yp6ynF7<bFhb5w4EkE?p~k?w{NL(P!;ErgT<T`}))5_QRN5T$8&0)jF|tx~l)L>Oru4Mx<-*)?M8B<FD2MMRgrEAFJ{TQat=!yqmOc
wK`v?+AqDz>v!?aS-Z^Q+=!Rs@><h#joCO6>G-p}Np@ail~<99$Ju_6F|+46oxh{F1jPX=uOzKg%*NYQouBG_gcKga>J6*79?LsV#hWX=Bg8{_BiBcrGp_3A
R{O!ybFtF%AyatOuKqYxzeUYkSNs&T50KyMAwQ6gt5>*usvgYR)ys^_)w;?_<}IqOu(}UZeJ{Ip(FYl)y8XbaZhD&Uul6Ic=ejz-)~%C#DLdUm`*~MAq>9_S
JS|qQP^}}U`L|V^BBi^->Opnz|J8XTX?~gNuUtFVr?-z&I7&56t?J3S{O=WyQ$6=8Uu{eccTsr(t{*r>yTR_;lU!GH?o;?yYA$VRj^Y`z*Qy>?8oyWLqtttI
<J{f(ReC<KtIMnMuv+MTtMvy~zhieDH8oGk#pzhT?&`^;>tk%(n$o4+#b;0P$lT|z#;a;SQ1u+8`uJ0MEUSaLOaHSvPavJ|_wfF5;}2Iq*6p*Z=68K&_5*tE
0rH2Hj+DxuPx0xhb5Gnp;NAU6X`Hsow@`V`)js!D+|MiCEXyxL`LgT(S^Ig;j6YKCwVL;)b<^4XRGnAD&f9hCx9<E|R#%6OH`wb|^8_{zsLm}<?SJ(?(*16%
A7<-csGnY4kClFHBd@3C`{{mwFO`3(`LNp;?c#KY>A%?diLcIU*z>8z+3wuWT|6yz?v0ASsQsC0{ZZw~rF1+{odV6tJ4yfU?uTl;&i1{mevhqd4)T3g`y&uP
C!I&BxKCgBHMd{<)%>RF=RM8%9`UQHb5u~iSJnS=>qn{nt#mx&>Md9Ko@zfOI#=wvs&l&4dR-d7PR&WG>R49eOBPp7)zOYkKfl|l^gMQ#C*94u!u{N>^1BBa
x47S{{*lE^+_}FlzsA)qPt6<HRnJ@1@2h+VR{z-5)q9wE?XG^tDlW$Ay}N!R6(6biojTux?UzCNXo%nSH1WyRd12SPlh=235RvX?wa--Tw@ud(R2+@v3Ap&(
BV-p;ecEc?fcSWW#D^~Oyt;8fnm178rLFRflnyM5KUMoM2e~f?z(H2uv$&euSMKgBm0yAEqB|d_>VH!@=qa4U)gN^EAFBStSzMnqf44dhi^b1WzGYR%K&>kv
yTkh7zImTl^@Vr$|J2X>p8-%y0Rj{Q6aWAK2mk;8Apq3Ec_ea~0|3C%0{{#F0000000000004ji00000MqhJdVQgh&E^csnP)h*<6ay3h000O8001EXDWmqr
KOg`AfQJJB3IG5A0000000000fB~GD0{}-~ZDnC@E^csnP)h*<6ay3h000O8001EX=-u}kXA}ScfQJJB2><{90000000000fB^%#0{}-~b9HSlZg6=}O928D
0~7!N00;m803iS$J|tf%3;+P2M*sj000000000000001h0g=uF0BvP$Vr6o3XlZaRZg6=}O928D0~7!N00;m803iT_{nH%i000000ssIC00000000000001h
0q5ES0Ap`;Zgg`lZg6=}O928D0~7!N00;m803iUuWfl0<0ssI^Jpcd-00000000000001h0r%Sj07o%ja&LDoZg6=}O928D0~7!N00;m803iUZX`@a<9RL7K
Jpcd-00000000000001h0rcJj07o%jV{dFOZg6=}O928D0~7!N00;m803iUO8N$Cc2mk;WdH?_n00000000000001h0bl$B07o%jc42IFWiD=Tc~DCM0u%!j
000080000X0Q<w*8m1cn01Wd0015yA00000000000Du9$1Oxy}PhWCxcP?&lc~DCM0u%!j000080000X05d^g*sLr701Wd0015yA00000000000Du9EAOrwQ
PhVqiY%Xqac~DCM0u%!j000080000X0QEPqScn<`0GQ<g01N;C00000000000Du8tO#}c-PhWOnY;|QWZg6=}O9ci10000B01E(d0ssIKXaoQN00'''

REDUCTION_PROVENANCE = {'algorithm': 'deterministic spherical Lloyd clustering of nonzero incoming DAN profiles', 'initialization': 'largest original column L2 norm then farthest-first cosine; source-index ties', 'profile_count': 96, 'nonzero_profile_count': 95, 'zero_profile_index': 95, 'zero_source_column_count': 2700, 'nonzero_source_column_count': 1852, 'iterations_executed': 7, 'iteration_limit': 12, 'mean_reconstruction_global_relative_frobenius_error': 0.8394112853488672, 'mean_reconstruction_nonzero_destination_relative_error_median': 0.914933574926541, 'mean_reconstruction_nonzero_destination_relative_error_p90': 2.2464836364966656, 'mean_reconstruction_nonzero_destination_relative_error_max': 4.415341266511408, 'unit_direction_destination_l2_error_median': 0.9879059389528136, 'unit_direction_destination_l2_error_p90': 1.164099713688666, 'max_DAN_routing_mass_residual': 4.440892098500626e-16, 'original_routing_mass': 122.63196371230326, 'group_sum_routing_mass': 122.63196371230326, 'nonzero_group_size_min': 1, 'nonzero_group_size_median': 13.0, 'nonzero_group_size_max': 186, 'zero_DAN_row_count': 41, 'empty_groups': [], 'v20_source_sha256': 'f0dad8a713d2f2889f65685ae1f1e53295c65fe8debcbfbd95dcb4fd48881e87', 'v20_F_max_absolute_differences': {'F1': 0.0, 'F2': 1.734723475976807e-18, 'F3': 4.336808689942018e-19, 'F4': 4.336808689942018e-19}, 'hemisphere_definition': 'Dominant DAN delivery side, not an anatomical profile hemisphere; mixed-source profiles remain mixed.', 'nonzero_profiles_mixed_at_one_percent': 82, 'nonzero_profile_dominant_side_purity_min': 0.5008431667959222, 'nonzero_profile_dominant_side_purity_median': 0.8128731732042803, 'source_nonzero_KCs_mixed_at_one_percent': 706, 'source_zip_name': 'loop_analysis_results.zip', 'source_zip_sha256': '13789f8373ac42bf9a43cf1e3b971c5b4fa04f72ae389a4a7eacc4e0b0be4c9b', 'source_member': 'loop_analysis_results/loop_operators.npz', 'source_member_sha256': 'ac213819b19ddf6f2a8184f33215ae6fa2722993e51772c2e838d8a9a5203868', 'source_G1_sha256': 'b8669c709a9227b2fc6f348a8c7dd68cad2daf451aa086dcbb0cf9c9359da45e', 'source_F_sha256': {'F1': '3d866ed6a4e2043313a320f36bdee307134162b091cd1fc44537cf8947dc424e', 'F2': '6d37a8d23ae71f09635da5a477039025970773492143f05b0e0d61bd5346f463', 'F3': 'f84f94b91dd95226988157e8762b5d5c41abb94e08c67bbb5b41b283e7df1e22', 'F4': '31118876a6ce8202d410368293645366d27700c82cb2bac15c8629aa6d3142ba'}, 'source_G1_shape': [301, 4552], 'source_F_shape': [104, 301], 'source_G1_nonzero_entries': 3891, 'source_index_order': 'DAN and KC numeric indices preserve archive row and column order; object ID arrays are not loaded', 'semantics': {'G1_sum': 'Exact sum of original destination columns by group; preserves routing mass for every DAN input.', 'G1_count_mean': 'G1_sum/group_counts; one mean incoming delivery profile per computational coordinate; used for ranking.', 'source_KC_group': 'Membership of every original KC destination; last group represents the identically zero direct profile.', 'zero_profile_assumption': 'The 2700 zero columns are one pattern, assigned one of 96 model coordinates; this is functional pattern reduction and does not preserve cell population frequencies.', 'limitations': 'Static count-weighted unsigned routing patterns. No exact plasticity sites, synaptic signs, activity gains, or persistence mechanism is inferred.'}, 'elapsed_seconds': 0.8386338210002577, 'teaching_row_geometry_diagnostics_nonzero_rows': {'count_mean': {'row_cosine_gram_relative_frobenius_error': 1.5384729610544032, 'off_diagonal_cosine_pearson_correlation': 0.5498839187275502, 'off_diagonal_cosine_absolute_error_median': 0.0, 'off_diagonal_cosine_absolute_error_p90': 0.04906979117922112}, 'sum': {'row_cosine_gram_relative_frobenius_error': 1.729709270883319, 'off_diagonal_cosine_pearson_correlation': 0.5118339783235026, 'off_diagonal_cosine_absolute_error_median': 0.0, 'off_diagonal_cosine_absolute_error_p90': 0.03268602252303067}, 'population_weighted_mean': {'row_cosine_gram_relative_frobenius_error': 1.6367569860398725, 'off_diagonal_cosine_pearson_correlation': 0.5353977271896737, 'off_diagonal_cosine_absolute_error_median': 0.0, 'off_diagonal_cosine_absolute_error_p90': 0.04175391139380668}}, 'scramble_definition': 'Independent permutation of all 95 live profiles in each teaching row; explicit zero profile fixed. Row multiset preserved, column marginals not preserved. Mixed profiles are not treated as anatomical hemispheres.', 'joint_readout_addressing': {'method': 'Retain raw direct DAN-to-KC routing and KC-to-MBON counts in common original cell axis order; form within-KC products before the existing 96-group mean.', 'v21_source_sha256': '1b8d47ddb21f654fc7df9ba879c7c316e5b6e1a31b0d1c6e9ebb8240989931b8', 'v21_payload_sha256': '07f1b94ab51696cf7d933b59b83af9cf30555ecb46cfecca84e7866915eb865d', 'KC_to_MBON_source_file': 'fly_distillation/results/KC_to_output_synapse_counts.npz', 'KC_to_MBON_source_file_sha256': '3e285dde1bc998c2c0239ab25a6a2c85c35dde839bd6307be672d80c12fa222c', 'DAN_to_KC_source_file': 'fly_distillation/results/teaching_to_KC_synapse_counts.npz', 'DAN_to_KC_source_file_sha256': 'e9342118ba2db717507d6a7d43bfb05087641ec262c6ddfa206c7d8188549c00', 'independent_raw_audit_file': 'claude_architecture_review/audit_blocks.npz', 'independent_raw_audit_file_sha256': '2fb5c1e6837929968d6c0a880d5090be5a3c2c5ea05505efb287b6d873879595', 'KC_to_MBON_raw_array_sha256': '7034ea7c5f51a789f18c332a3963fba0031a95b55394b587d731051ff6d063cf', 'DAN_to_KC_raw_array_sha256': 'da9ed5580c232f2e91df502f1ccb9403a5cb3683fbe72f50af2855cf316c5d53', 'axis_provenance_file': 'fly_distillation/results/interface_provenance.csv', 'axis_provenance_file_sha256': 'ddd8ea819c98370b1dd6c4e5535068a884c499dfe681cfbd3ec23bfe41b81bd1', 'axis_validation': {'KC_to_MBON_dense_sparse_exact_equal': True, 'DAN_to_KC_dense_sparse_exact_equal': True, 'G1_raw_DAN_to_KC_support_exact_equal': True, 'within_DAN_max_normalization_ratio_range': 6.938893903907228e-18, 'G1_raw_DAN_to_KC_scaled_max_absolute_error': 1.3877787807814457e-17, 'G1_raw_DAN_to_KC_scaled_relative_frobenius_error': 7.453905684489353e-18, 'meaning': 'The saved sparse blocks and independent raw audit agree entrywise in KC/DAN/MBON axis order. G1 has identical DK support and a constant count-to-routing normalization within each nonzero DAN row. No object ID arrays are needed at runtime.'}, 'KC_to_MBON_shape': [4552, 104], 'KC_to_MBON_nonzero_entries': 15523, 'KC_to_MBON_raw_synapse_count_sum': 28415.0, 'KC_to_MBON_covered_KCs': 3587, 'directly_addressed_KCs': 1852, 'directly_addressed_KCs_missing_MBON_target': 28, 'lost_raw_G1_total_routing_mass_fraction': 0.010265761153445607, 'lost_pooled_G_mean_routing_mass_fraction': 0.013146345407563563, 'covered_joint_MBON_marginal_vs_G_mean_relative_frobenius_error': 0.05747460525923696, 'MBON_columns_without_joint_support': 3, 'old_payload_arrays_preserved_exactly': True, 'runtime_sparse_schema': {'G1': 'G1_row, G1_col, G1_value; shape (301,4552)', 'KO': 'KO_row, KO_col, KO_value; raw unsigned counts, shape (4552,104)'}, 'normalization': 'KO rows are normalized over measured MBON destinations by the controller. Empty rows stay zero. Original G1 values and original 96-group assignment are preserved.', 'engineering_assumptions': ['Whole-KC co-targeting approximates potential shared teaching/readout territory; it is not a map of exact synaptic modulation sites.', 'Squared fixed task code participation transfers MBON priorities to task readout heads without asserting a biological task correspondence.'], 'null': 'Permute KO-derived normalized readout profiles only within each original G1 group, preserving G1 and each group readout-profile multiset while disrupting within-KC pairing.'}}

def load_anatomy():
    """Decode verified numeric source data; no pickle, scipy, or raw fly files."""
    raw = base64.b85decode(''.join(ANATOMY_B85.split()))
    if sha_bytes(raw) != ANATOMY_SHA256:
        raise ValueError('Embedded anatomy checksum mismatch; download the script again.')
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        a = {name: archive[name].copy() for name in archive.files}

    def require(condition, message):
        if not condition:
            raise ValueError('Invalid embedded anatomy: ' + message)

    expected = {'F_scaled', 'G_mean', 'G_sum', 'membership', 'counts',
                'G1_row', 'G1_col', 'G1_value', 'KO_row', 'KO_col', 'KO_value'}
    require(set(a) == expected, 'unexpected array schema')
    for name, shape in (('F_scaled', (104, 301)), ('G_mean', (301, 96)),
                        ('G_sum', (301, 96)), ('membership', (4552,)), ('counts', (96,))):
        require(a[name].shape == shape, name + ' shape')
        require(np.issubdtype(a[name].dtype, np.number), name + ' numeric dtype')
        require(np.isfinite(a[name]).all() and (a[name] >= 0).all(), name + ' values')
    for name in ('membership', 'counts'):
        require(np.issubdtype(a[name].dtype, np.integer), name + ' integer dtype')
    require((a['membership'] < 96).all(), 'membership range')
    require(((a['counts'] > 0) & (a['counts'] <= 4552)).all(), 'group counts')
    require(np.array_equal(np.bincount(a['membership'], minlength=96), a['counts']),
            'membership counts mismatch')
    require(np.allclose(a['G_sum'], a['G_mean'] * a['counts'][None, :],
                        rtol=1e-12, atol=1e-14), 'mean/sum grouping mismatch')
    for prefix, shape, expected_nnz in (('G1', (301, 4552), 3891),
                                       ('KO', (4552, 104), 15523)):
        row, col, value = (a[prefix + suffix] for suffix in ('_row', '_col', '_value'))
        require(row.ndim == col.ndim == value.ndim == 1, prefix + ' sparse dimensions')
        require(len(row) == len(col) == len(value) == expected_nnz, prefix + ' sparse size')
        require(np.issubdtype(row.dtype, np.integer) and np.issubdtype(col.dtype, np.integer),
                prefix + ' index dtype')
        require(((row >= 0) & (row < shape[0])).all() and
                ((col >= 0) & (col < shape[1])).all(), prefix + ' sparse index range')
        require(np.issubdtype(value.dtype, np.number), prefix + ' value dtype')
        require(np.isfinite(value).all() and (value > 0).all(), prefix + ' finite positive values')
        flat = row.astype(np.int64) * shape[1] + col
        require(len(np.unique(flat)) == len(flat), prefix + ' duplicate sparse entries')
    require(np.equal(a['KO_value'], np.floor(a['KO_value'])).all(), 'KO integral counts')
    require(float(a['KO_value'].sum()) == 28415.0, 'KO raw synapse count sum')
    reconstructed = np.zeros_like(a['G_sum'])
    np.add.at(reconstructed, (a['G1_row'], a['membership'][a['G1_col']]), a['G1_value'])
    require(np.allclose(reconstructed, a['G_sum'], rtol=1e-12, atol=1e-14),
            'original G1 aggregation mismatch')
    return a


def dependencies(install):
    import importlib.util
    missing = [name for name in ('numpy', 'torch') if importlib.util.find_spec(name) is None]
    if missing and not install:
        raise SystemExit('Missing '+', '.join(missing)+'. Run again with --install-deps.')
    if missing:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', *missing])
    global np, torch, nn, F
    import numpy as np
    import torch
    from torch import nn
    from torch.nn import functional as F


def seed_for(seed, name):
    return int.from_bytes(hashlib.sha256(f'{seed}:{name}'.encode()).digest()[:4], 'little')


def rng_for(seed, name):
    return np.random.default_rng(seed_for(seed, name))


def sha_bytes(x):
    return hashlib.sha256(x).hexdigest()


def make_elm(cfg, seed):
    class ScalarELM(nn.Module):
        """v17 ungated ordinary-cell arithmetic, adapted to a small cue stream.

        All cells communicate one activity value and have six private states.
        Sensory topology and delay assignment are new task-specific draws;
        this is not a claim to reproduce v17 SHD results or load its weights.
        """
        def __init__(self):
            super().__init__()
            N, R, M, H, K = cfg.neurons, 8, 6, 12, 4
            self.N=N; self.R=R; self.M=M; self.K=K
            torch.manual_seed(seed_for(seed, 'elm_parameters'))
            def normal(shape, sd):
                return nn.Parameter(torch.randn(*shape)*sd)
            def xavier(shape):
                x=torch.empty(*shape)
                for block in x: nn.init.xavier_uniform_(block)
                return nn.Parameter(x)
            self.syn=normal((N,R,4),.2)
            self.bias=nn.Parameter(torch.full((N,R),-.05))
            self.gain=nn.Parameter(torch.full((N,R),math.log(math.expm1(1.))))
            self.w1=xavier((K,R+M,H));self.b1=nn.Parameter(torch.zeros(K,H))
            self.w2=xavier((K,H,M));self.b2=nn.Parameter(torch.zeros(K,M))
            self.wm=normal((N,M),.3);self.wd=normal((N,R),.1)
            self.abias=nn.Parameter(torch.zeros(N))
            self.threshold=nn.Parameter(torch.full((N,),math.log(math.expm1(.75))))
            rg=rng_for(seed,'elm_tau')
            lt=np.tile(np.linspace(math.log(1.5),math.log(32.),M),(N,1))
            lt=np.clip(lt+rg.normal(0,.05,lt.shape),math.log(1.5),math.log(32.))
            u=np.clip((lt-math.log(1.5))/math.log(32./1.5),1e-4,1-1e-4)
            self.tau_raw=nn.Parameter(torch.tensor(np.log(u/(1-u)),dtype=torch.float32))
            rg=rng_for(seed,'elm_topology')
            sensory=np.stack([rg.choice(32,R*3,replace=False).reshape(R,3) for _ in range(N)])
            recurrent=np.stack([rg.choice(np.delete(np.arange(N),i),R,replace=False) for i in range(N)])
            self.register_buffer('sensory',torch.tensor(sensory,dtype=torch.long))
            self.register_buffer('recurrent',torch.tensor(recurrent,dtype=torch.long))
            self.register_buffer('delays',torch.tensor(rg.integers(1,4,size=(N,R)),dtype=torch.long))
            self.register_buffer('type_index',torch.arange(N)%K)

        def forward(self, x):
            B,T,D=x.shape
            assert D==32
            # These resets occur for EVERY independent experience and probe.
            m=x.new_zeros(B,self.N,6)
            history=[x.new_zeros(B,self.N) for _ in range(3)]
            tau=torch.exp(math.log(1.5)+math.log(32./1.5)*torch.sigmoid(self.tau_raw))
            alpha=-torch.expm1(-1./tau)[None]
            gain=F.softplus(self.gain)[None];thr=F.softplus(self.threshold)[None]
            w1=self.w1[self.type_index];w2=self.w2[self.type_index]
            b1=self.b1[self.type_index];b2=self.b2[self.type_index]
            histidx=((self.delays-1)*self.N+self.recurrent).flatten()
            for t in range(T):
                sensed=x[:,t].index_select(1,self.sensory.flatten()).reshape(B,self.N,8,3)
                rec=torch.stack(history,1).flatten(1).index_select(1,histidx).reshape(B,self.N,8)
                gathered=torch.cat([sensed,rec.unsqueeze(-1)],-1)
                branch=torch.tanh(gain*((gathered*self.syn).sum(-1)+self.bias))
                z=torch.cat([branch,m],-1)
                hidden=torch.tanh(torch.einsum('bnq,nqh->bnh',z,w1)+b1)
                candidate=torch.tanh(torch.einsum('bnh,nhm->bnm',hidden,w2)+b2)
                m=(1-alpha)*m+alpha*candidate
                v=(m*self.wm).sum(-1)+(branch*self.wd).sum(-1)+self.abias
                activity=torch.sigmoid(4*(v-thr))
                history=[activity]+history[:-1]
            return activity
    return ScalarELM()


def cue_bases(rng, count):
    return rng.normal(0,.65,size=(count,2,32)).astype(np.float32)


def render_cues(bases, variants, rng, noise):
    """Two ordered pulse patterns followed by blanks; ID is never an input."""
    B=len(bases); x=np.zeros((B,variants,12,32),np.float32)
    for i in range(B):
        for v in range(variants):
            first=bases[i,0]+rng.normal(0,noise,32)
            second=bases[i,1]+rng.normal(0,noise,32)
            x[i,v,0:3]=first
            x[i,v,5:8]=second
    return x


def encode(model, x, batch_size=32):
    model.eval();result=[]
    with torch.no_grad():
        for i in range(0,len(x),batch_size):
            result.append(model(torch.from_numpy(x[i:i+batch_size])).cpu().numpy())
    return np.concatenate(result)


def develop_encoder(cfg, seed):
    """Small separate self-supervised development, charged in exposure ledger.

    No A/B/revision cues or outcome labels enter development or calibration.
    Predict a fixed random projection of both ordered pulse patterns.
    """
    model=make_elm(cfg,seed)
    rg=rng_for(seed,'development_data')
    bases=cue_bases(rg,cfg.dev_cues)
    x=render_cues(bases,1,rg,cfg.noise)[:,0]
    proj=rng_for(seed,'development_target').normal(size=(64,16))/8.
    y=(bases.reshape(len(bases),64)@proj).astype(np.float32)
    torch.manual_seed(seed_for(seed,'development_head'))
    head=nn.Linear(cfg.neurons,16)
    optimizer=torch.optim.AdamW(list(model.parameters())+list(head.parameters()),lr=.003,weight_decay=1e-4)
    tx=torch.from_numpy(x);ty=torch.from_numpy(y)
    order_rng=rng_for(seed,'development_order')
    losses=[]
    for epoch in range(cfg.dev_passes):
        model.train();loss_sum=0.
        for indices in np.array_split(order_rng.permutation(len(x)),math.ceil(len(x)/16)):
            optimizer.zero_grad(set_to_none=True)
            a=model(tx[indices]); pred=head(a)
            loss=F.mse_loss(pred,ty[indices])+.001*(a.mean()-.15).square()
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.)
            optimizer.step();loss_sum+=float(loss.detach())*len(indices)
        losses.append(loss_sum/len(x))
    for p in model.parameters():p.requires_grad_(False)
    # Calibration is an additional forward exposure of development inputs.
    calibration=encode(model,x)
    mean=calibration.mean(0);sd=np.maximum(calibration.std(0),.01)
    state_hash=sha_bytes(b''.join(t.detach().cpu().numpy().tobytes() for t in model.state_dict().values()))
    info={'loss_by_pass':losses,'development_unique_inputs':len(x),
          'development_training_presentations':len(x)*cfg.dev_passes,
          'development_calibration_presentations':len(x),
          'outcome_labels_used':0,'encoder_sha256':state_hash,
          'encoder_parameters':sum(p.numel() for p in model.parameters()),
          'temporary_development_head_parameters':sum(p.numel() for p in head.parameters()),
          'final_activity_std_median':float(np.median(calibration.std(0)))}
    return model,mean,sd,info


def sparse_features(activity, mean, sd, fraction):
    standardized=np.clip((activity-mean)/sd,-5,5).astype(np.float64)
    k=max(1,round(standardized.shape[1]*fraction))
    chosen=np.argpartition(np.abs(standardized),-k,axis=1)[:,-k:]
    result=np.zeros_like(standardized)
    np.put_along_axis(result,chosen,np.take_along_axis(standardized,chosen,axis=1),axis=1)
    result/=np.maximum(np.linalg.norm(result,axis=1,keepdims=True),1e-12)
    # Signed standardized features are an engineering representation choice;
    # they are not claimed to be biological firing rates or recorded KC sparsity.
    return result


def make_codebook(seed):
    # Equal-energy orthogonal label codes; seed varies across replications.
    # No biological cell is silently identified with one of the four task labels.
    q,_=np.linalg.qr(rng_for(seed,'task_output_codes').normal(size=(104,4)))
    return (q.T*math.sqrt(104)).astype(np.float64)


def assess(learner,features,labels,masks):
    before=learner.digest()
    prediction=learner.predict(features.reshape(-1,features.shape[-1])).reshape(features.shape[:2])
    assert before==learner.digest(),'Evaluation modified durable memory'
    correct=(prediction==labels[:,None])
    result={}
    for name,mask in masks.items():
        result[name]=float(correct[mask].mean()) if np.any(mask) else None
    return result


def write_csv(path,rows):
    if not rows:return
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)




class Route:
    """Fixed whole-cell co-targeting transferred to four computational readouts.

    G1 and KC->MBON contacts are combined at each original KC before pooling.
    The code participation C**2/104 is an engineering transfer, not a claim
    that biological readouts correspond to the task's four outcome labels.
    Only the fixed codebook enters construction; learning data do not.
    """
    def __init__(self, anatomy, seed, codebook):
        self.F = np.asarray(anatomy['F_scaled'], dtype=np.float64)
        self.codes = np.asarray(codebook, dtype=np.float64)
        if self.codes.shape != (4, 104):
            raise ValueError('The fixed pilot requires a 4 by 104 codebook.')
        if not np.allclose(self.codes @ self.codes.T, 104*np.eye(4), atol=1e-10):
            raise ValueError('The codebook must have orthogonal equal-energy rows.')
        group = np.asarray(anatomy['membership'], dtype=np.int64)
        counts = np.asarray(anatomy['counts'], dtype=np.float64)
        base = np.asarray(anatomy['G_mean'], dtype=np.float64)
        rows = np.asarray(anatomy['G1_row'], dtype=np.int64)
        cells = np.asarray(anatomy['G1_col'], dtype=np.int64)
        values = np.asarray(anatomy['G1_value'], dtype=np.float64)
        if group.shape != (4552,) or counts.shape != (96,):
            raise ValueError('Anatomical axes do not match the fixed 96-profile pilot.')
        if not np.array_equal(np.bincount(group, minlength=96), counts):
            raise ValueError('Group membership and counts disagree.')
        A = np.zeros((4552, 104), dtype=np.float64)
        np.add.at(A, (anatomy['KO_row'], anatomy['KO_col']), anatomy['KO_value'])
        if not np.isfinite(A).all() or (A < 0).any():
            raise ValueError('KC-to-MBON source counts must be finite and nonnegative.')
        row_mass = A.sum(axis=1)
        A /= np.where(row_mass > 0, row_mass, 1.0)[:, None]
        participation = self.codes.T**2 / 104.0
        kc_task = A @ participation

        # Each group's multiset of complete KC readout profiles is preserved.
        # G1, KC memberships and the G1-only predecessor are held exactly fixed.
        rg = rng_for(seed, 'v22_within_group_KC_readout_shuffle')
        permutation = np.arange(4552)
        for index in range(96):
            members = np.flatnonzero(group == index)
            permutation[members] = rg.permutation(members)
        kc_task_null = kc_task[permutation]
        pooled = np.zeros((301, 96, 4), dtype=np.float64)
        null = np.zeros_like(pooled)
        source_weight = values / counts[group[cells]]
        for head in range(4):
            np.add.at(pooled[:, :, head], (rows, group[cells]),
                      source_weight * kc_task[cells, head])
            np.add.at(null[:, :, head], (rows, group[cells]),
                      source_weight * kc_task_null[cells, head])
        self.assignment = rng_for(seed, 'v21_profile_to_elm_assignment').permutation(96)
        self.tie_rank = rng_for(seed, 'v21_address_tie_rank').permutation(96)
        self.G = base[:, self.assignment].copy()
        self.H = pooled[:, self.assignment, :].copy()
        self.H_scrambled = null[:, self.assignment, :].copy()
        self.live = self.G.sum(axis=0) > 0
        self.within_group_permutation = permutation
        self.group_membership = group.copy()
        self.kc_readout_profiles = A

        g_cell_mass = np.bincount(cells, weights=values, minlength=4552)
        unsupported = (g_cell_mass > 0) & (row_mass == 0)
        surviving = np.zeros_like(base)
        np.add.at(surviving, (rows, group[cells]),
                  source_weight * (row_mass[cells] > 0))
        self._coverage = {
            'directly_addressed_KCs': int(np.count_nonzero(g_cell_mass)),
            'directly_addressed_KCs_without_readout': int(unsupported.sum()),
            'raw_G1_mass_lost_to_missing_readout_fraction':
                float(g_cell_mass[unsupported].sum()/max(values.sum(), 1e-30)),
            'pooled_G1_mass_lost_to_missing_readout_fraction':
                float((base.sum()-surviving.sum())/max(base.sum(), 1e-30)),
            'pooled_G1_relative_coverage_discrepancy':
                float(np.linalg.norm(base-surviving)/max(np.linalg.norm(base), 1e-30)),
            'KC_groups_of_size_one': int(np.count_nonzero(counts == 1)),
            'within_group_shuffle_preserves_KC_groups':
                bool(np.array_equal(group[permutation], group)),
            'within_group_shuffle_preserves_full_profile_multisets':
                bool(np.array_equal(np.sort(permutation), np.arange(4552))
                     and np.array_equal(group[permutation], group)),
            'KC_readout_profiles_are_row_normalized':
                bool(np.allclose(A[row_mass > 0].sum(axis=1), 1.0)),
        }
        if not all(np.isfinite(x).all() and (x >= 0).all()
                   for x in (self.F, self.G, self.H, self.H_scrambled)):
            raise ValueError('Routing operators must be finite and nonnegative.')

    def scores(self, prediction, target, arm, cut_feedback=False):
        if arm == 'uniform_delta':
            return np.zeros((96, 4), dtype=np.float64)
        if arm not in ('g1_feedback', 'joint_feedback', 'joint_scrambled',
                       'joint_outcome_only'):
            raise ValueError('Unknown local-learning arm: '+str(arm))
        residual = target if cut_feedback or arm == 'joint_outcome_only' else target-prediction
        teaching = np.abs((residual @ self.codes) @ self.F)
        if arm == 'g1_feedback':
            return np.repeat((teaching @ self.G)[:, None], 4, axis=1)
        delivery = self.H_scrambled if arm == 'joint_scrambled' else self.H
        return (teaching @ delivery.reshape(301, -1)).reshape(96, 4)

    def diagnostics(self):
        return {
            'assignment_sha256': sha_bytes(self.assignment.astype('<i8').tobytes()),
            'measured_delivery_sha256': sha_bytes(self.G.tobytes()),
            'joint_delivery_sha256': sha_bytes(self.H.tobytes()),
            'joint_scrambled_delivery_sha256': sha_bytes(self.H_scrambled.tobytes()),
            'within_group_KC_permutation_sha256':
                sha_bytes(self.within_group_permutation.astype('<i8').tobytes()),
            'joint_scrambled_relative_operator_difference':
                float(np.linalg.norm(self.H-self.H_scrambled)/max(np.linalg.norm(self.H), 1e-30)),
            'measured_zero_feature_profiles': int((self.G.sum(axis=0) == 0).sum()),
            'joint_zero_feature_head_profiles': int((self.H.sum(axis=0) == 0).sum()),
            'scrambled_zero_feature_head_profiles': int((self.H_scrambled.sum(axis=0) == 0).sum()),
            'G1_predecessor_unchanged_by_joint_null': True,
            'joint_tensor_marginals_preserved_by_null': False,
            'mapping_uses_observed_task_labels': False,
            'mapping_uses_fixed_random_task_codebook': True,
            'profile_assignment_is_anatomical_cell_identity': False,
            **self._coverage,
        }


def select_mask(h, scores, energy_floor, tie_rank, uniform=False):
    """V21's exact feature-energy selection, used independently per readout."""
    energy = h*h
    active = np.flatnonzero(energy > 0)
    total = float(energy.sum())
    mask = np.zeros(len(h), dtype=bool)
    if total == 0:
        return mask, 0.0, 'zero_feature'
    if uniform:
        mask[active] = True
        return mask, total, 'uniform_control'
    values = scores[active]
    tolerance = 1e-12*max(1.0, float(np.max(np.abs(values))))
    if float(np.ptp(values)) <= tolerance:
        mask[active] = True
        return mask, total, 'uninformative_scores'
    order = active[np.lexsort((tie_rank[active], -values))]
    cumulative = np.cumsum(energy[order])
    count = min(len(order), int(np.searchsorted(cumulative, energy_floor*total, side='left'))+1)
    mask[order[:count]] = True
    selected = float(energy[mask].sum())
    while selected < energy_floor*total and count < len(order):
        mask[order[count]] = True
        count += 1
        selected = float(energy[mask].sum())
    return mask, selected, 'addressed'


def head_mask_diagnostics(mask):
    distinct = len({mask[:, head].tobytes() for head in range(4)})
    equal, jaccard = [], []
    for first in range(4):
        for second in range(first+1, 4):
            a, b = mask[:, first], mask[:, second]
            equal.append(float(np.array_equal(a, b)))
            union = int(np.count_nonzero(a | b))
            jaccard.append(float(np.count_nonzero(a & b)/union) if union else 1.0)
    return {
        'distinct_readout_masks': float(distinct),
        'readout_mask_pair_equality': float(np.mean(equal)),
        'readout_mask_pair_jaccard': float(np.mean(jaccard)),
    }


class LocalLearner:
    """V is the complete durable memory; no old probes or task phase are inputs.

    Four-head storage exactly factors V21's W=V@C for its shared-mask arms.
    Joint masks change write placement without adding latent output directions.
    propose() is pure: counterfactual updates are inspected by the evaluator
    without changing memory or passing any of its old probes into this class.
    """
    def __init__(self, arm, route, codebook, cfg):
        self.arm = arm
        self.route = route
        self.codes = codebook
        self.eta = float(cfg.learning_rate)
        self.floor = float(cfg.write_energy_floor)
        if not 0 < self.floor <= 1 or not 0 < self.eta <= 1:
            raise ValueError('Energy floor and learning rate must lie in (0, 1].')
        self.V = np.zeros((96, 4), dtype=np.float64)

    def output(self, h):
        return h @ self.V

    def predict(self, h):
        return np.argmax(self.output(h), axis=-1)

    def digest(self):
        return sha_bytes(self.V.tobytes())

    def propose(self, h, label, mode=None, cut_feedback=False):
        mode = self.arm if mode is None else mode
        h = np.asarray(h, dtype=np.float64)
        if h.shape != (96,) or not np.isfinite(h).all():
            raise ValueError('Expected one finite 96-coordinate feature vector.')
        if not isinstance(label, (int, np.integer)) or not 0 <= label < 4:
            raise ValueError('Outcome must be an integer from 0 to 3.')
        prediction = self.output(h)
        target = np.eye(4)[label]
        delta = target-prediction
        scores = self.route.scores(prediction, target, mode, cut_feedback)
        mask = np.zeros((96, 4), dtype=bool)
        selected = np.zeros(4, dtype=np.float64)
        reason = []
        for head in range(4):
            mask[:, head], selected[head], why = select_mask(
                h, scores[:, head], self.floor, self.route.tie_rank,
                uniform=mode == 'uniform_delta')
            reason.append(why)
        denominators = np.where(selected > 0, selected, 1.0)
        change = self.eta*(h[:, None]*mask)*delta[None, :]/denominators[None, :]
        total = float(h@h)
        desired = self.eta*delta if total > 0 else np.zeros(4)
        correction_error = float(np.linalg.norm(h@change-desired)/max(np.linalg.norm(desired), 1e-12))
        if correction_error > 1e-10:
            raise AssertionError('A readout failed the matched current-example correction.')
        if total and np.any(selected < self.floor*total*(1-1e-12)):
            raise AssertionError('A readout failed the retained-energy bound.')
        ordinary = self.eta*np.outer(h, delta)/total if total else np.zeros_like(change)
        norm = float(np.linalg.norm(change))
        baseline_norm = float(np.linalg.norm(ordinary))
        if norm > baseline_norm/math.sqrt(self.floor)+1e-12:
            raise AssertionError('The selective-write norm bound was violated.')
        return {
            'change': change, 'mask': mask, 'selected_energy': selected,
            'reason': reason, 'scores': scores, 'delta': delta,
            'pre_output': prediction.copy(), 'mode': mode,
            'cut_feedback': bool(cut_feedback), 'label': int(label),
            'state_digest': self.digest(), 'input_digest': sha_bytes(h.tobytes()),
            'correction_relative_error': correction_error,
        }

    def observe(self, h, label, proposal=None):
        h = np.asarray(h, dtype=np.float64)
        proposal = self.propose(h, label) if proposal is None else proposal
        if (proposal['state_digest'] != self.digest()
                or proposal['input_digest'] != sha_bytes(h.tobytes())
                or proposal['label'] != label):
            raise ValueError('Cannot apply a stale proposal or one for another example.')
        if proposal['mode'] != self.arm or proposal['cut_feedback']:
            raise ValueError('Counterfactual proposals cannot replace the configured policy.')
        delta = proposal['delta']
        change = proposal['change']
        mask = proposal['mask']
        selected = proposal['selected_energy']
        reasons = proposal['reason']
        prediction = int(np.argmax(proposal['pre_output']))
        total = float(h@h)
        active = h != 0
        nactive = int(active.sum())
        counterfactual_available = self.arm in ('g1_feedback', 'joint_feedback', 'joint_scrambled')
        changed, jaccard, changed_fraction = 0.0, 1.0, 0.0
        if counterfactual_available:
            cut = self.propose(h, label, cut_feedback=True)['mask']
            changed = float(not np.array_equal(mask, cut))
            changed_fraction = float(np.mean(np.any(mask != cut, axis=0)))
            per_head_jaccard = []
            for head in range(4):
                union = int(np.count_nonzero(mask[:, head] | cut[:, head]))
                per_head_jaccard.append(
                    float(np.count_nonzero(mask[:, head] & cut[:, head])/union) if union else 1.0)
            jaccard = float(np.mean(per_head_jaccard))
        ordinary_norm = self.eta*float(np.linalg.norm(delta))/math.sqrt(total) if total else 0.0
        norm = float(np.linalg.norm(change))
        qactive = proposal['scores'][active]
        qcv = [float(np.std(qactive[:, head])/max(float(np.mean(qactive[:, head])), 1e-30))
               for head in range(4)] if nactive else [0.0]*4
        after_delta = delta-h@change
        self.V += change
        if not np.isfinite(self.V).all():
            raise FloatingPointError('Nonfinite associative memory.')
        info = {
            'pre_update_mse': float(np.mean(delta**2)),
            'current_label_mse_improvement': float(np.mean(delta**2)-np.mean(after_delta**2)),
            'selected_fraction_of_active_features': float(mask.sum()/(4*nactive)) if nactive else 0.0,
            'retained_feature_energy_fraction': float(np.mean(selected)/total) if total else 0.0,
            'minimum_readout_retained_energy_fraction': float(selected.min()/total) if total else 0.0,
            'selected_features_without_direct_teaching':
                float(np.count_nonzero(mask & ~self.route.live[:, None])/4),
            'update_l2': norm,
            'same_state_update_norm_ratio': norm/ordinary_norm if ordinary_norm else 0.0,
            'current_cue_correction_relative_error': proposal['correction_relative_error'],
            'score_coefficient_of_variation': float(np.mean(qcv)),
            'uninformative_score_fallback': float(np.mean([r == 'uninformative_scores' for r in reasons])),
            'zero_feature_no_update': float(total == 0),
            'learned_feedback_counterfactual_available': float(counterfactual_available),
            'mask_changed_when_feedback_cut': changed,
            'head_mask_changed_fraction_when_feedback_cut': changed_fraction,
            'mask_jaccard_when_feedback_cut': jaccard,
            'local_normalization_applicable': 1.0,
            **head_mask_diagnostics(mask),
        }
        return prediction, info


class CumulativeRidge:
    """Factorized V21 cumulative-statistics reference; covariance memory counted.

    This objective keeps conflicting old and revised targets equally weighted.
    It has no forgetting factor, replay buffer or outcome-dependent routing.
    """
    def __init__(self, features, codebook, ridge):
        if ridge <= 0:
            raise ValueError('Ridge regularization must be positive.')
        self.codes = codebook
        self.V = np.zeros((features, 4), dtype=np.float64)
        self.inverse = np.eye(features, dtype=np.float64)/ridge

    def output(self, h):
        return h @ self.V

    def predict(self, h):
        return np.argmax(self.output(h), axis=-1)

    def digest(self):
        return sha_bytes(self.V.tobytes()+self.inverse.tobytes())

    def observe(self, h, label):
        z = self.output(h)
        prediction = int(np.argmax(z))
        delta = np.eye(4)[label]-z
        a = self.inverse @ h
        denom = 1+float(h@a)
        if not np.isfinite(denom) or denom <= 0:
            raise FloatingPointError('Cumulative ridge lost positive definiteness.')
        change = np.outer(a/denom, delta)
        self.V += change
        self.inverse -= np.outer(a, a)/denom
        self.inverse = (self.inverse+self.inverse.T)*0.5
        if not np.isfinite(self.V).all() or not np.isfinite(self.inverse).all():
            raise FloatingPointError('Nonfinite cumulative-ridge memory.')
        after = delta-h@change
        live = bool(np.any(h))
        return prediction, {
            'pre_update_mse': float(np.mean(delta**2)),
            'current_label_mse_improvement': float(np.mean(delta**2)-np.mean(after**2)),
            # Ridge writes through inverse@h, including inactive coordinates;
            # mask/energy diagnostics do not apply (flagged explicitly below).
            'selected_fraction_of_active_features': 0.0,
            'retained_feature_energy_fraction': 0.0,
            'minimum_readout_retained_energy_fraction': 0.0,
            'selected_features_without_direct_teaching': 0.0,
            'update_l2': float(np.linalg.norm(change)),
            'same_state_update_norm_ratio': 0.0,
            'current_cue_correction_relative_error': 0.0,
            'score_coefficient_of_variation': 0.0,
            'uninformative_score_fallback': 0.0,
            'zero_feature_no_update': float(not live),
            'learned_feedback_counterfactual_available': 0.0,
            'mask_changed_when_feedback_cut': 0.0,
            'head_mask_changed_fraction_when_feedback_cut': 0.0,
            'mask_jaccard_when_feedback_cut': 1.0,
            'local_normalization_applicable': 0.0,
            'distinct_readout_masks': 0.0,
            'readout_mask_pair_equality': 0.0,
            'readout_mask_pair_jaccard': 0.0,
        }


def check_v20_stream_compatibility(cfg,costs):
    expected = {
        17: ('81608cc2292c651a8e012ceef434f8f3c7e279ed5ede6249ca056aac8a2c5642',
             '121a8e7072d1b3bfc1161639fd88d61ab9f24a74d41a6bc1026cd27623b4b969',
             '29dc4f527f362be2a023445ef6c54d972c310827cfffa6b0783f022f8df61301'),
        29: ('0eb666278a77656a3676c242ebc62e052524022830eb1ef181a82a9b16cbd1a7',
             '2ea8b6afcb7ab5c0b325f0a1319476b04753ce4a7064a296b34b44d6080d84b0',
             'a10c95de6cf076f543982e893e922ead4860020c457a366621503fd02bf60e81'),
        43: ('061736e429d2bf8b9395e9aec1f4b1e41b19935dea0d70ac1debabcb7b48bd1b',
             'd2eecfd4a70bd1f0a29efea0ea5d49337229b5451a667020ef1269edcc1adfae',
             '2c1ac33ec1a8e4a1e14c810f3921753c158ebc2c4b8390754fed9b0fc48f59c2')}
    applicable = cfg.group_cues == 24 and cfg.variants == 2 and cfg.probe_variants == 3 and cfg.passes == 3 and cfg.reverse_cues == 12 and cfg.noise == .06
    result = []
    for c in costs:
        if not applicable or c['seed'] not in expected:
            result.append({'seed':c['seed'],'status':'not_applicable_for_this_stream_configuration'})
            continue
        actual = tuple(c[key] for key in ('stream_sha256','online_unique_training_inputs_sha256','heldout_probe_inputs_sha256'))
        matched = actual == expected[c['seed']]
        result.append({'seed':c['seed'],'status':'matched' if matched else 'mismatch',
                       'note':'Compares data and labeled order, not hardware-dependent encoder weights.'})
        if not matched:
            raise AssertionError('Default stream no longer matches the archived v20 experiment')
    return result






# Shared online-learning interface. This file is assembled into the deliverable.
class MemoryBase:
    def __init__(self, d, seed):
        self.d = int(d)
        self.rng = np.random.default_rng(int(seed))
        self.t = 0

    def mutable_arrays(self):
        return {k: v for k, v in vars(self).items() if isinstance(v, np.ndarray)}

    def diagnostics(self):
        return {}

    def reminded_output(self, H, reminders):
        return self.output(H)

    def state_digest(self):
        digest = hashlib.sha256()
        for k, v in sorted(self.mutable_arrays().items()):
            digest.update(k.encode()); digest.update(np.asarray(v).tobytes())
        scalars = {k: v for k, v in vars(self).items()
                   if isinstance(v, (str, int, float, bool, type(None)))}
        digest.update(json.dumps(scalars, sort_keys=True, allow_nan=False).encode())
        digest.update(json.dumps(self.rng.bit_generator.state, sort_keys=True).encode())
        return digest.hexdigest()

    def mutable_bytes(self):
        return int(sum(v.nbytes for v in self.mutable_arrays().values()))


def unit_rows(x):
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def target4(label):
    y = np.zeros(4, dtype=np.float64); y[int(label)] = 1.0
    return y


def delta_change(h, error, eta=.6):
    return eta*np.outer(h, error)/max(float(h@h), 1e-12)


def bounded_change(change, limit=2.0):
    norm = float(np.linalg.norm(change))
    return change * min(1.0, limit/max(norm, 1e-12))


class PlainMemory(MemoryBase):
    def __init__(self, d, seed, eta=.6):
        super().__init__(d, seed); self.V = np.zeros((d, 4)); self.eta = float(eta)

    def output(self, H):
        return np.asarray(H)@self.V

    def observe(self, h, label):
        change = delta_change(h, target4(label)-self.output(h), self.eta)
        self.V += change; self.t += 1
        return {'extra_labeled_presentations': 0, 'update_norm': float(np.linalg.norm(change))}


# Online weight-memory candidates. All are explicitly engineering representatives:
# Importance is an activity/confidence usage trace, not a reproduction of SI/EWC;
# Bayes is a diagonal Kalman approximation, not MESU; projection is a bounded
# past-input subspace, not task-boundary OGD; consolidation is a small coupled
# state model; cascade is a discrete signed-synapse proxy, not exact Fusi dynamics.
# No class receives cue identities, phase boundaries, future labels, or probes.


class ImportanceMemory(MemoryBase):
    """Causal usage protection, optionally released by confident contradiction.

    strength is 3 (moderate) or 12 (strong). Each successful write adds feature
    energy to a neuron or feature/output usage trace. Future writes are divided
    by 1+strength*usage, with the ORIGINAL feature-energy denominator: this
    deliberately reduces the achieved correction rather than renormalizing away
    protection. decay is per external event. scramble applies a fixed independent
    row permutation to each output's historical protection, preserving its values.
    anatomical additionally restricts each output using the fixed v22 routing.
    """
    def __init__(self, d, seed, granularity='pair', strength=3., eta=.6,
                 decay=0., release=False, scramble=False, anatomical=False,
                 route=None):
        super().__init__(d, seed)
        if granularity not in ('pair', 'neuron'):
            raise ValueError('importance granularity must be pair or neuron')
        self.V = np.zeros((d, 4))
        self.importance = np.zeros((d, 4 if granularity == 'pair' else 1))
        self.granularity = granularity
        self.strength = float({'moderate': 3., 'strong': 12.}.get(strength, strength))
        self.eta = float(eta); self.decay = float(decay)
        self.release = bool(release); self.scramble = bool(scramble)
        self.anatomical = bool(anatomical); self.route = route
        if self.anatomical and route is None:
            raise ValueError('anatomical importance requires embedded routing')
        self.permutation = (np.column_stack([self.rng.permutation(d)
                                           for _ in range(self.importance.shape[1])])
                            if self.scramble else None)
        self.surprise_ema = 1.; self.releases = 0
        self.gain_sum = 0.; self.correction_sum = 0.

    def output(self, H):
        return np.asarray(H) @ self.V

    def observe(self, h, label):
        h = np.asarray(h); y = target4(label); prediction = self.output(h)
        error = y-prediction; surprise = float(error@error)
        self.importance *= 1.-self.decay
        # A novel all-zero readout is not a confident contradiction. Release
        # requires an existing competing score, and only touches active features.
        margin = float(np.max(np.delete(prediction, int(label)))-prediction[int(label)])
        conflict = self.release and margin > .30 and surprise > 1.25*self.surprise_ema
        if conflict:
            active = np.abs(h) > 1e-12
            self.importance[active] *= .35
            self.releases += 1
        used = self.importance
        if self.scramble:
            used = np.take_along_axis(used, self.permutation, axis=0)
        gains = 1./(1.+self.strength*used)
        mask = np.ones((self.d, 4))
        if self.anatomical:
            scores = self.route.scores(prediction, y, 'joint_feedback')
            for j in range(4):
                mask[:, j] = select_mask(h, scores[:, j], .5, self.route.tie_rank)[0]
        unprotected = delta_change(h, error, self.eta)
        change = bounded_change(unprotected*gains*mask)
        self.V += change
        post_error = y-self.output(h)
        reliability = float(np.clip(1.-float(post_error@post_error)/2., .05, 1.))
        if self.granularity == 'pair':
            credit = np.outer(h*h, .10+y)
        else:
            credit = (.35*h*h)[:, None]
        # Trace the locations actually eligible for learning in the anatomy arm.
        trace_mask = mask if self.granularity == 'pair' else np.mean(mask, axis=1, keepdims=True)
        self.importance += reliability*credit*trace_mask
        np.minimum(self.importance, 1e6, out=self.importance)
        self.surprise_ema = .95*self.surprise_ema+.05*surprise
        active = np.abs(h) > 1e-12
        self.gain_sum += float(np.mean(gains[active])) if np.any(active) else 1.
        self.correction_sum += float(np.linalg.norm(h@change))/max(float(np.linalg.norm(error)), 1e-12)
        self.t += 1
        return {'extra_labeled_presentations': 0, 'update_norm': float(np.linalg.norm(change)),
                'release': float(conflict)}

    def diagnostics(self):
        return {'importance_mean': float(np.mean(self.importance)),
                'mean_plasticity_gain': self.gain_sum/max(self.t, 1),
                'mean_correction_fraction': self.correction_sum/max(self.t, 1),
                'release_count': self.releases}


class DiagonalBayesMemory(MemoryBase):
    """Diagonal Kalman proxy. process_noise reopens uncertainty every event.

    noise=.25 gives an initial gain of .8 for unit-norm features. Keeping only
    the posterior diagonal is a deliberate approximation and is not MESU.
    """
    def __init__(self, d, seed, process_noise=0., noise=.25):
        super().__init__(d, seed)
        # The four output posteriors have identical covariance under the common
        # feature/noise model; storing one shared diagonal avoids duplicated state.
        self.V = np.zeros((d, 4)); self.variance = np.ones(d)
        self.process_noise = float(process_noise); self.noise = float(noise)
        if self.process_noise < 0. or self.noise <= 0.:
            raise ValueError('Bayesian proxy requires nonnegative process noise and positive noise')
        self.gain_sum = 0.

    def output(self, H):
        return np.asarray(H)@self.V

    def observe(self, h, label):
        h = np.asarray(h); error = target4(label)-self.output(h)
        prior = np.minimum(self.variance+self.process_noise, 1.)
        denominator = self.noise+float(np.sum(h*h*prior))
        gain = prior*h/denominator
        change = np.outer(gain, error)
        self.V += change
        self.variance[:] = np.clip(prior*(1.-gain*h), 1e-8, 1.)
        self.gain_sum += float(h@gain)
        self.t += 1
        return {'extra_labeled_presentations': 0, 'update_norm': float(np.linalg.norm(change))}

    def diagnostics(self):
        return {'posterior_variance_mean': float(np.mean(self.variance)),
                'posterior_variance_min': float(np.min(self.variance)),
                'mean_current_input_gain': self.gain_sum/max(self.t, 1)}


class ProjectionMemory(MemoryBase):
    """Bounded online input subspace; no task boundaries or outcome memory.

    The current error is projected using the PREVIOUS basis, then the input is
    admitted to that basis. leak=0 protects its span exactly; positive leak lets
    revision modify protected directions. Crucially the denominator remains h'h,
    so a nearly-null available direction is never blown up to a full correction.
    With turnover=0 a full basis is frozen. Positive turnover replaces one basis
    vector every specified number of novel events, explicitly sacrificing some
    old protection. rank counts allocated columns, including unfilled columns.
    """
    def __init__(self, d, seed, rank=16, leak=0., eta=.6, turnover=0):
        super().__init__(d, seed)
        self.rank = min(max(1, int(rank)), d)
        self.V = np.zeros((d, 4)); self.basis = np.zeros((d, self.rank))
        self.filled = 0; self.leak = float(leak); self.eta = float(eta)
        if not 0. <= self.leak <= 1.:
            raise ValueError('projection leak must lie between zero and one')
        self.turnover = max(0, int(turnover)); self.novel_count = 0; self.replace_index = 0
        self.available_energy_sum = 0.; self.null_updates = 0

    def output(self, H):
        return np.asarray(H)@self.V

    def observe(self, h, label):
        h = np.asarray(h); error = target4(label)-self.output(h)
        Q = self.basis[:, :self.filled]
        residual = h-Q@(Q.T@h)
        # A second Gram-Schmidt pass prevents accumulated numerical leakage.
        residual -= Q@(Q.T@residual)
        direction = self.leak*h+(1.-self.leak)*residual
        change = self.eta*np.outer(direction, error)/max(float(h@h), 1e-12)
        self.V += change
        energy = float(direction@direction)/max(float(h@h), 1e-12)
        self.available_energy_sum += energy
        self.null_updates += int(energy < 1e-16)
        residual_norm = float(np.linalg.norm(residual))
        if residual_norm > 1e-7:
            self.novel_count += 1
            if self.filled < self.rank:
                self.basis[:, self.filled] = residual/residual_norm
                self.filled += 1
            elif self.turnover and self.novel_count % self.turnover == 0:
                self.basis[:, self.replace_index] = residual/residual_norm
                self.replace_index = (self.replace_index+1) % self.rank
        self.t += 1
        return {'extra_labeled_presentations': 0, 'update_norm': float(np.linalg.norm(change))}

    def diagnostics(self):
        Q = self.basis[:, :self.filled]
        return {'subspace_rank': self.filled, 'mean_available_energy': self.available_energy_sum/max(self.t, 1),
                'null_update_fraction': self.null_updates/max(self.t, 1),
                'basis_orthogonality_error': float(np.linalg.norm(Q.T@Q-np.eye(self.filled)))}


class ConsolidationMemory(MemoryBase):
    """Two/three bidirectionally coupled synaptic states, first state expressed.

    Capacities [1,8] or [1,4,16] give hidden states slower motion. Neighbor fluxes
    are evaluated simultaneously and conserve capacity-weighted state. Their
    unequal equilibration means the expressed state is not an ordinary delta
    rule in disguise. feedback reduces coupling after surprising external events;
    it is an error-gated engineering rule, not measured biological consolidation.
    Replay, when supplied, changes plasticity before ONE external coupling tick.
    """
    def __init__(self, d, seed, states=2, feedback=False, eta=.6, coupling=.04):
        super().__init__(d, seed)
        if int(states) not in (2, 3):
            raise ValueError('consolidation states must be 2 or 3')
        self.states = int(states); self.levels = np.zeros((self.states, d, 4))
        self.capacities = np.array([1., 8.] if self.states == 2 else [1., 4., 16.])
        self.feedback = bool(feedback); self.eta = float(eta); self.coupling = float(coupling)
        if not 0. <= self.coupling <= .25:
            raise ValueError('coupling must be between zero and .25 for stable explicit flux')
        self.gate_sum = 0.; self.replay_presentations = 0

    def output(self, H):
        return np.asarray(H)@self.levels[0]

    def plasticity(self, h, label):
        change = bounded_change(delta_change(h, target4(label)-self.output(h), self.eta))
        self.levels[0] += change
        return float(np.linalg.norm(change))

    def plastic_update(self, h, label):
        return self.plasticity(h, label)

    def advance_clock(self, surprise=None):
        gate = 1.
        if self.feedback:
            gate = .15+.85*math.exp(-max(float(surprise or 0.), 0.)/.5)
        flux_change = np.zeros_like(self.levels)
        for j in range(self.states-1):
            flux = self.coupling*gate*(.25**j)*(self.levels[j]-self.levels[j+1])
            flux_change[j] -= flux/self.capacities[j]
            flux_change[j+1] += flux/self.capacities[j+1]
        self.levels += flux_change
        self.gate_sum += gate; self.t += 1
        return float(np.linalg.norm(flux_change[0]))

    def observe_with_replay(self, h, label, replay_pair=None):
        before = self.levels[0].copy()
        error = target4(label)-self.output(h)
        surprise = float(error@error)
        self.plasticity(h, label)
        extra = int(replay_pair is not None)
        if replay_pair is not None:
            self.plasticity(replay_pair[0], replay_pair[1])
            self.replay_presentations += 1
        self.advance_clock(surprise)
        return {'extra_labeled_presentations': extra,
                'update_norm': float(np.linalg.norm(self.levels[0]-before))}

    def observe(self, h, label):
        return self.observe_with_replay(h, label)

    def diagnostics(self):
        return {'expressed_weight_norm': float(np.linalg.norm(self.levels[0])),
                'hidden_weight_norm': float(np.linalg.norm(self.levels[1:])),
                'state_disagreement': float(np.linalg.norm(self.levels[0]-self.levels[-1])),
                'mean_consolidation_gate': self.gate_sum/max(self.t, 1),
                'replay_presentations': self.replay_presentations}


class CascadeMemory(MemoryBase):
    """Discrete metaplastic engineering proxy with signed expressed weights.

    Repeated agreeing local error proposals deepen a synapse. A conflicting
    proposal has probability proportional to 2**(-depth); an accepted conflict
    first reduces depth, and reverses sign only at zero depth. This is a bounded
    local transition rule inspired by cascade ideas, not an exact Fusi model.
    Zero signifies an initially uncommitted synapse. amplitude fixes weight size.
    """
    def __init__(self, d, seed, levels=4, amplitude=.20, rate=.6):
        super().__init__(d, seed)
        self.levels = int(levels); self.amplitude = float(amplitude); self.rate = float(rate)
        if not 2 <= self.levels <= 32:
            raise ValueError('cascade levels must be between 2 and 32')
        self.signs = np.zeros((d, 4), dtype=np.int8)
        self.depth = np.zeros((d, 4), dtype=np.int8)
        self.sign_changes = 0; self.depth_changes = 0

    def output(self, H):
        return np.asarray(H)@(self.amplitude*self.signs)

    def observe(self, h, label):
        error = target4(label)-self.output(h)
        gradient = np.outer(h, error)
        proposal = np.sign(gradient).astype(np.int8)
        probability = np.clip(self.rate*math.sqrt(self.d)*np.abs(gradient), 0., 1.)
        random_values = self.rng.random((self.d, 4))
        old_signs = self.signs.copy(); old_depth = self.depth.copy()
        active = proposal != 0
        uncommitted = active & (self.signs == 0) & (random_values < probability)
        agrees = active & (self.signs == proposal) & (random_values < .5*probability)
        opposes = active & (self.signs != 0) & (self.signs != proposal)
        accepted = opposes & (random_values < probability*np.exp2(-self.depth.astype(float)))
        flip = accepted & (self.depth == 0)
        shallow = accepted & (self.depth > 0)
        self.signs[uncommitted | flip] = proposal[uncommitted | flip]
        self.depth[agrees] = np.minimum(self.depth[agrees]+1, self.levels-1)
        self.depth[shallow] -= 1
        self.depth[uncommitted | flip] = 0
        self.sign_changes += int(np.count_nonzero(self.signs != old_signs))
        self.depth_changes += int(np.count_nonzero(self.depth != old_depth))
        self.t += 1
        return {'extra_labeled_presentations': 0,
                'update_norm': float(self.amplitude*np.linalg.norm(self.signs.astype(float)-old_signs))}

    def diagnostics(self):
        return {'mean_depth': float(np.mean(self.depth)),
                'deepest_fraction': float(np.mean(self.depth == self.levels-1)),
                'committed_fraction': float(np.mean(self.signs != 0)),
                'sign_changes': self.sign_changes, 'depth_changes': self.depth_changes}


class ActiveForgettingMemory(MemoryBase):
    """Two learners with different decay/rates, mixed by causal recent errors.

    Both learners see each external label once. They decay on the external clock.
    The readout mixture is inferred from pre-update prediction errors, without
    task labels or phase switches. This is an active-forgetting ensemble proxy.
    """
    def __init__(self, d, seed, fast_decay=.01, slow_decay=.0001,
                 fast_eta=.8, slow_eta=.15, temperature=.1):
        super().__init__(d, seed)
        self.weights = np.zeros((2, d, 4)); self.losses = np.ones(2)
        self.decays = np.array([fast_decay, slow_decay], dtype=float)
        self.rates = np.array([fast_eta, slow_eta], dtype=float)
        self.temperature = float(temperature)
        self.fast_mixture_sum = 0.

    def mixture(self):
        logits = -(self.losses-np.min(self.losses))/max(self.temperature, 1e-8)
        p = np.exp(logits); return p/np.sum(p)

    def output(self, H):
        combined = np.tensordot(self.mixture(), self.weights, axes=(0, 0))
        return np.asarray(H)@combined

    def observe(self, h, label):
        before = np.tensordot(self.mixture(), self.weights, axes=(0, 0))
        y = target4(label)
        predictions = np.einsum('d,kdc->kc', h, self.weights)
        observed_losses = np.mean((predictions-y[None, :])**2, axis=1)
        self.losses[:] = .95*self.losses+.05*observed_losses
        for j in range(2):
            self.weights[j] *= 1.-self.decays[j]
            self.weights[j] += bounded_change(delta_change(h, y-h@self.weights[j], self.rates[j]))
        after = np.tensordot(self.mixture(), self.weights, axes=(0, 0))
        self.fast_mixture_sum += float(self.mixture()[0]); self.t += 1
        return {'extra_labeled_presentations': 0, 'update_norm': float(np.linalg.norm(after-before))}

    def diagnostics(self):
        return {'fast_mixture_final': float(self.mixture()[0]),
                'fast_mixture_mean': self.fast_mixture_sum/max(self.t, 1),
                'fast_recent_error': float(self.losses[0]), 'slow_recent_error': float(self.losses[1]),
                'learner_disagreement': float(np.linalg.norm(self.weights[0]-self.weights[1]))}


class SurpriseMemory(MemoryBase):
    """Immediate or exponentially smoothed surprise changes learning rate only.

    surprise is squared label prediction error divided by a past error EMA.
    Rates stay between .15*eta and 1.5*eta (at most .95). It stores no exemplar
    and never accesses future outcomes or an externally supplied novelty flag.
    """
    def __init__(self, d, seed, mode='immediate', eta=.6):
        super().__init__(d, seed)
        if mode not in ('immediate', 'smoothed'):
            raise ValueError('surprise mode must be immediate or smoothed')
        self.V = np.zeros((d, 4)); self.mode = mode; self.eta = float(eta)
        self.error_ema = 1.; self.surprise_ema = 1.; self.rate_sum = 0.
        self.rate_min = 1.; self.rate_max = 0.

    def output(self, H):
        return np.asarray(H)@self.V

    def observe(self, h, label):
        error = target4(label)-self.output(h)
        error_energy = float(error@error)
        normalized = min(error_energy/max(self.error_ema, .05), 5.)
        self.surprise_ema = .90*self.surprise_ema+.10*normalized
        signal = normalized if self.mode == 'immediate' else self.surprise_ema
        rate = min(.95, self.eta*float(np.clip(signal, .15, 1.5)))
        change = bounded_change(delta_change(h, error, rate)); self.V += change
        self.error_ema = .95*self.error_ema+.05*error_energy
        self.rate_sum += rate; self.rate_min = min(self.rate_min, rate)
        self.rate_max = max(self.rate_max, rate); self.t += 1
        return {'extra_labeled_presentations': 0, 'update_norm': float(np.linalg.norm(change))}

    def diagnostics(self):
        return {'mean_learning_rate': self.rate_sum/max(self.t, 1),
                'minimum_learning_rate': self.rate_min, 'maximum_learning_rate': self.rate_max,
                'normalized_surprise_ema': self.surprise_ema}


class FastWeightsMemory(MemoryBase):
    """Slow delta-rule weights plus a decaying fast associative matrix.

    Hebbian fast writes store h*target, residual writes store h*current_error.
    Fast decay occurs once per external event. Both matrices are exposed in the
    resource accounting. This is a fast-association proxy, not a transformer.
    """
    def __init__(self, d, seed, mode='residual', decay=.03, fast_eta=.6, slow_eta=.15):
        super().__init__(d, seed)
        if mode not in ('hebbian', 'residual'):
            raise ValueError('fastweights mode must be hebbian or residual')
        self.V = np.zeros((d, 4)); self.fast = np.zeros((d, 4))
        self.mode = mode; self.decay = float(decay)
        self.fast_eta = float(fast_eta); self.slow_eta = float(slow_eta)
        self.fast_clips = 0

    def output(self, H):
        return np.asarray(H)@(self.V+self.fast)

    def observe(self, h, label):
        before = self.V+self.fast; y = target4(label)
        error = y-np.asarray(h)@before
        self.fast *= 1.-self.decay
        self.V += bounded_change(delta_change(h, error, self.slow_eta))
        target = y if self.mode == 'hebbian' else error
        self.fast += bounded_change(delta_change(h, target, self.fast_eta))
        fast_norm = float(np.linalg.norm(self.fast))
        if fast_norm > 20.:
            self.fast *= 20./fast_norm; self.fast_clips += 1
        self.t += 1
        return {'extra_labeled_presentations': 0,
                'update_norm': float(np.linalg.norm(self.V+self.fast-before))}

    def diagnostics(self):
        return {'slow_weight_norm': float(np.linalg.norm(self.V)),
                'fast_weight_norm': float(np.linalg.norm(self.fast)), 'fast_norm_clips': self.fast_clips}


def build_weight_memory(spec, d, seed, route=None):
    kind = spec['kind']; params = dict(spec.get('params', {}))
    if kind == 'importance':
        return ImportanceMemory(d, seed, route=route, **params)
    constructors = {'bayes': DiagonalBayesMemory, 'projection': ProjectionMemory,
                    'consolidation': ConsolidationMemory, 'cascade': CascadeMemory,
                    'active_forgetting': ActiveForgettingMemory, 'surprise': SurpriseMemory,
                    'fastweights': FastWeightsMemory}
    return constructors[kind](d, seed, **params) if kind in constructors else None


def weight_mechanism_tests():
    """Small arithmetic/causality checks; no benchmark fitting or model selection."""
    d = 8; a = np.eye(d)[0]; b = np.eye(d)[1]; mixed = (a+b)/math.sqrt(2.)
    result = {}
    strict = ProjectionMemory(d, 1, rank=2, leak=0.)
    strict.observe(a, 0); remembered = strict.output(a).copy()
    strict.observe(mixed, 1)
    assert np.allclose(strict.output(a), remembered, atol=1e-12)
    strict.observe(a, 2)
    assert np.allclose(strict.output(a), remembered, atol=1e-12)
    leaky = ProjectionMemory(d, 1, rank=2, leak=.15)
    leaky.observe(a, 0); old = leaky.output(a).copy(); leaky.observe(a, 2)
    assert not np.allclose(leaky.output(a), old)
    result['projection_protects_past_span_and_leak_allows_revision'] = True

    bayes = DiagonalBayesMemory(d, 1)
    bayes.observe(a, 0)
    assert np.all(bayes.variance[0] < 1.) and np.all(bayes.variance[1] == 1.)
    forgetful = DiagonalBayesMemory(d, 1, process_noise=.003)
    for _ in range(30):
        bayes.observe(a, 0); forgetful.observe(a, 0)
    assert np.mean(forgetful.variance[0]) > np.mean(bayes.variance[0])
    result['diagonal_uncertainty_contracts_and_process_noise_reopens'] = True

    protected = ImportanceMemory(d, 1, strength=3.)
    for _ in range(8):
        protected.observe(a, 0)
    matched = PlainMemory(d, 1); matched.V[:] = protected.V
    pnorm = protected.observe(a, 1)['update_norm']
    nnorm = matched.observe(a, 1)['update_norm']
    assert pnorm < nnorm
    result['importance_suppresses_same_state_write_without_renormalizing'] = True

    consolidation = ConsolidationMemory(d, 1, states=3)
    consolidation.observe(a, 0)
    assert float(np.linalg.norm(consolidation.levels[1])) > 0.
    weighted = np.einsum('s,sdc->dc', consolidation.capacities, consolidation.levels)
    consolidation.advance_clock()
    assert np.allclose(weighted, np.einsum('s,sdc->dc', consolidation.capacities, consolidation.levels))
    before_t = consolidation.t
    report = consolidation.observe_with_replay(a, 0, (b, 1))
    assert consolidation.t == before_t+1 and report['extra_labeled_presentations'] == 1
    result['consolidation_conserves_flux_and_replay_uses_one_clock'] = True

    models = [ImportanceMemory(d, 1), ImportanceMemory(d, 1, granularity='neuron', release=True),
              ImportanceMemory(d, 1, scramble=True), DiagonalBayesMemory(d, 1),
              ProjectionMemory(d, 1, rank=3, turnover=2), ConsolidationMemory(d, 1, feedback=True),
              CascadeMemory(d, 1, levels=4), ActiveForgettingMemory(d, 1),
              SurpriseMemory(d, 1), SurpriseMemory(d, 1, mode='smoothed'),
              FastWeightsMemory(d, 1, mode='hebbian'), FastWeightsMemory(d, 1)]
    rng = np.random.default_rng(223)
    H = unit_rows(rng.normal(size=(40, d)))
    for model in models:
        for i, h in enumerate(H):
            report = model.observe(h, i % 4)
            assert report['extra_labeled_presentations'] == 0
            assert np.all(np.isfinite(model.output(H[:3])))
        assert model.t == len(H)
        digest = model.state_digest()
        assert model.output(H[:3]).shape == (3, 4)
        assert model.output(H[0]).shape == (4,)
        assert model.state_digest() == digest
        assert all(np.all(np.isfinite(x)) for x in model.mutable_arrays().values())
        assert all(np.isfinite(v) for v in model.diagnostics().values())
        if isinstance(model, CascadeMemory):
            assert np.min(model.depth) >= 0 and np.max(model.depth) < model.levels
            assert set(np.unique(model.signs)).issubset({-1, 0, 1})
    result['all_weight_families_finite_and_inference_pure'] = True
    assert not np.allclose(models[-1].output(H), models[-2].output(H))
    result['hebbian_and_residual_fast_writes_are_distinct'] = True
    return result


# Structural engineering proxies, deliberately not full paper reproductions.

def _structural_spec(spec):
    if isinstance(spec, dict):
        return spec.get('kind'), dict(spec.get('params', {}))
    return spec.kind, dict(spec.params)


def _structural_rows(H):
    H = np.asarray(H, dtype=np.float64)
    return (H[None, :] if H.ndim == 1 else H), H.ndim == 1


def _structural_softmax(x):
    x = np.asarray(x, dtype=np.float64)
    ex = np.exp(x-np.max(x, axis=-1, keepdims=True))
    return ex/np.maximum(ex.sum(axis=-1, keepdims=True), 1e-12)


class SparseSpecializationMemory(MemoryBase):
    """Positive expansion and competitive local specialization; no momentum.

    Signed frozen ELM features become a positive 2d-dimensional input code.
    Nonnegative normalized addresses compete; only winners receive local
    prototype adaptation. This is an SDM/FlyModel-inspired engineering proxy.
    Frozen controls have identical initial addresses, width, and output rules.
    """
    def __init__(self, d, seed, width=96, k=8, adaptive=True, eta=.6,
                 address_eta=.015, **unused):
        super().__init__(d, seed)
        self.width = int(width); self.k = min(int(k), self.width)
        self.adaptive = bool(adaptive); self.eta = float(eta)
        self.address_eta = float(address_eta)
        # Sparse positive address vectors avoid an almost constant dense mean.
        address = self.rng.exponential(1.0, (self.width, 2*d))
        address *= self.rng.random(address.shape) < .18
        self.address = unit_rows(address)
        self.V = np.zeros((self.width, 4))
        self.usage = np.zeros(self.width)

    def _encode(self, H):
        H, scalar = _structural_rows(H)
        pos = unit_rows(np.concatenate((np.maximum(H, 0), np.maximum(-H, 0)), axis=1))
        activation = np.maximum(pos@self.address.T, 0)
        indices = np.argsort(activation, axis=1, kind='stable')[:, -self.k:]
        mask = np.zeros_like(activation, dtype=bool)
        np.put_along_axis(mask, indices, True, axis=1)
        code = unit_rows(activation*mask)
        return code, pos, mask, scalar

    def output(self, H):
        code, _, _, scalar = self._encode(H)
        result = code@self.V
        return result[0] if scalar else result

    def observe(self, h, label):
        code, pos, mask, _ = self._encode(h)
        u = code[0]; winners = mask[0] & (u > 0)
        before = self.V.copy()
        error = target4(label)-u@self.V
        self.V += delta_change(u, error, self.eta)
        # Explicit nonnegative bounded output state, including inactive units.
        np.clip(self.V, 0, 2, out=self.V)
        if self.adaptive and np.any(winners):
            self.address[winners] = unit_rows(
                (1-self.address_eta)*self.address[winners]
                + self.address_eta*pos[0])
        self.usage[winners] += 1
        self.t += 1
        return {'extra_labeled_presentations': 0,
                'update_norm': float(np.linalg.norm(self.V-before)),
                'active_units': int(winners.sum())}

    def mutable_arrays(self):
        arrays = {'V': self.V, 'usage': self.usage}
        if self.adaptive:
            arrays['address'] = self.address
        return arrays

    def diagnostics(self):
        return {'allocated_units': self.width,
                'unused_units': int(np.sum(self.usage == 0)),
                'mean_active_fraction': float(self.usage.sum()/max(1, self.t*self.width)),
                'fixed_bytes': 0 if self.adaptive else int(self.address.nbytes)}


class BoundedAllocationMemory(MemoryBase):
    """Causal match/reuse/allocate prototype memory with a hard slot bound.

    Prototype similarity selects matches. High-similarity conflicting labels
    revise that prototype, so revision does not require a task or cue identity.
    At capacity, novel inputs either recycle least-useful slots or are rejected.
    This is an ART/SQHN-inspired proxy, not either original algorithm.
    """
    def __init__(self, d, seed, slots=32, threshold=.90, recycle=True,
                 eta=.8, prototype_eta=.05, neighbors=1, quantized=False,
                 code_k=24, **unused):
        super().__init__(d, seed)
        self.slots = int(slots); self.threshold = float(threshold)
        self.recycle = bool(recycle); self.eta = float(eta)
        self.prototype_eta = float(prototype_eta)
        self.neighbors = max(1, int(neighbors))
        self.quantized = bool(quantized); self.code_k = min(int(code_k), d)
        self.centers = np.zeros((self.slots, 2*d if self.quantized else d))
        self.values = np.zeros((self.slots, 4))
        self.class_mass = np.zeros((self.slots, 4))
        self.utility = np.zeros(self.slots)
        self.counts = np.zeros(self.slots, dtype=np.int64)
        self.used = 0; self.allocations = 0; self.replacements = 0
        self.rejected = 0; self.revisions = 0

    def _content_code(self, H):
        H, scalar = _structural_rows(H)
        if not self.quantized:
            code = unit_rows(H)
        else:
            # Sparse quantized signed coordinates; no labels choose the code.
            index = np.argsort(np.abs(H), axis=1, kind='stable')[:, -self.code_k:]
            value = np.take_along_axis(H, index, axis=1)
            signed_index = index+(value < 0)*self.d
            code = np.zeros((len(H), 2*self.d))
            np.put_along_axis(code, signed_index, (value != 0).astype(float), axis=1)
            code = unit_rows(code)
        return code[0] if scalar else code

    def output(self, H):
        H, scalar = _structural_rows(H)
        result = np.zeros((len(H), 4))
        if self.used:
            similarities = self._content_code(H)@self.centers[:self.used].T
            n = min(self.neighbors, self.used)
            selected = np.argsort(similarities, axis=1, kind='stable')[:, -n:]
            similarities = np.take_along_axis(similarities, selected, axis=1)
            p = _structural_softmax(16*similarities)
            result = np.sum(p[:, :, None]*self.values[selected], axis=1)
        return result[0] if scalar else result

    def observe(self, h, label):
        h = self._content_code(h)
        target = target4(label); self.utility *= .999
        best = -1; similarity = -1.0
        if self.used:
            sims = self.centers[:self.used]@h
            best = int(np.argmax(sims)); similarity = float(sims[best])
        novel = similarity < self.threshold
        replacing = False
        if novel:
            if self.used < self.slots:
                best = self.used; self.used += 1; self.allocations += 1
            elif self.recycle:
                best = int(np.argmin(self.utility[:self.used]))
                self.replacements += 1; replacing = True
            else:
                self.rejected += 1; self.t += 1
                return {'extra_labeled_presentations': 0, 'update_norm': 0.0,
                        'capacity_rejection': 1}
            old_value = self.values[best].copy()
            self.centers[best] = h; self.values[best] = target
            self.class_mass[best] = target
            self.counts[best] = 1; self.utility[best] = 1
        else:
            old_value = self.values[best].copy()
            if int(np.argmax(old_value)) != int(label):
                self.revisions += 1
            if self.quantized:
                # Quantized addresses stay quantized; label evidence is local
                # and forgets obsolete contradictory evidence at fixed rate.
                self.class_mass[best] *= (1-self.eta)
                self.class_mass[best] += self.eta*target
                self.values[best] = self.class_mass[best]/max(
                    float(self.class_mass[best].sum()), 1e-12)
            else:
                self.centers[best] = unit_rows(
                    (1-self.prototype_eta)*self.centers[best]+self.prototype_eta*h)
                self.values[best] += self.eta*(target-self.values[best])
            self.counts[best] += 1; self.utility[best] += 1
        self.t += 1
        return {'extra_labeled_presentations': 0,
                'update_norm': float(np.linalg.norm(self.values[best]-old_value)),
                'allocated': int(novel and not replacing),
                'recycled': int(replacing), 'match_similarity': similarity}

    def diagnostics(self):
        return {'used_slots': self.used, 'slot_capacity': self.slots,
                'quantized_addresses': int(self.quantized),
                'total_allocations': self.allocations,
                'slot_replacements': self.replacements,
                'capacity_rejections': self.rejected,
                'conflicting_label_revisions': self.revisions}


class ContentBankMemory(MemoryBase):
    """Bounded local readout banks with content-inferred access.

    A recent-bank prior influences ordinary expression, but content retrieval
    remains active. Unlabeled reminders replace only the temporary access
    prior used by a pure prediction call; they never write storage or receive
    labels, cue IDs, phase IDs, or desired-output information. Null reminder
    benefit is a legitimate result when the stream offers no shared context.
    """
    def __init__(self, d, seed, banks=8, routing='inferred', eta=.6,
                 recent_bias=.35, novelty=.70, temperature=8.0, **unused):
        super().__init__(d, seed)
        self.banks = int(banks); self.routing = str(routing)
        self.eta = float(eta); self.recent_bias = float(recent_bias)
        self.novelty = float(novelty); self.temperature = float(temperature)
        self.V = np.zeros((self.banks, d, 4))
        self.centers = np.zeros((self.banks, d))
        self.counts = np.zeros(self.banks, dtype=np.int64)
        self.last_seen = np.zeros(self.banks, dtype=np.int64)
        self.fixed_address = unit_rows(self.rng.normal(size=(self.banks, d)))
        self.used = 0; self.active = -1; self.route_changes = 0

    def _content_probabilities(self, H):
        H = unit_rows(H)
        result = np.zeros((len(H), self.banks))
        if self.routing == 'fixed':
            # A matched-capacity random hash, independent of outcomes and time.
            index = np.argmax(H@self.fixed_address.T, axis=1)
            result[np.arange(len(H)), index] = 1.0
        elif self.used:
            result[:, :self.used] = _structural_softmax(
                self.temperature*(H@self.centers[:self.used].T))
        return result

    def _predict(self, H, reminders=None):
        H, scalar = _structural_rows(H)
        p = self._content_probabilities(H)
        prior = np.zeros(self.banks)
        if reminders is not None and len(reminders):
            reminders, _ = _structural_rows(reminders)
            prior = self._content_probabilities(reminders).mean(axis=0)
        elif self.active >= 0:
            prior[self.active] = 1.0
        if prior.sum() > 0:
            p = (1-self.recent_bias)*p+self.recent_bias*prior[None, :]
        values = np.einsum('nd,bdc->nbc', H, self.V, optimize=False)
        result = np.sum(p[:, :, None]*values, axis=1)
        return result[0] if scalar else result

    def output(self, H):
        return self._predict(H)

    def reminded_output(self, H, reminders):
        # Deliberately no mutation: counterfactual access to the same storage.
        return self._predict(H, reminders)

    def observe(self, h, label):
        h = np.asarray(h, dtype=np.float64)
        q = unit_rows(h); y = target4(label)
        if self.routing == 'fixed':
            bank = int(np.argmax(self.fixed_address@q))
            self.used = max(self.used, bank+1)
        elif not self.used:
            bank = 0; self.used = 1; self.centers[bank] = q
        else:
            similarities = self.centers[:self.used]@q
            bank = int(np.argmax(similarities))
            residual = float(np.linalg.norm(y-h@self.V[bank]))
            # Both conditions use information available on this experience.
            if (similarities[bank] < self.novelty and residual > .5
                    and self.used < self.banks):
                bank = self.used; self.used += 1; self.centers[bank] = q
        old = self.V[bank].copy()
        self.V[bank] += delta_change(h, y-h@self.V[bank], self.eta)
        self.counts[bank] += 1
        rate = min(.05, 1.0/float(self.counts[bank]))
        self.centers[bank] = unit_rows((1-rate)*self.centers[bank]+rate*q)
        self.route_changes += int(self.active >= 0 and bank != self.active)
        self.active = bank; self.t += 1; self.last_seen[bank] = self.t
        return {'extra_labeled_presentations': 0,
                'update_norm': float(np.linalg.norm(self.V[bank]-old)),
                'selected_bank': bank}

    def mutable_arrays(self):
        return {'V': self.V, 'centers': self.centers, 'counts': self.counts,
                'last_seen': self.last_seen}

    def diagnostics(self):
        return {'bank_capacity': self.banks,
                'used_banks': int(np.sum(self.counts > 0)),
                'route_changes': self.route_changes,
                'fixed_bytes': int(self.fixed_address.nbytes),
                'recent_access_weight': self.recent_bias}


class RecyclingAdapterMemory(MemoryBase):
    """Small trainable adapter after the frozen ELM, with optional turnover.

    Supervised gradients move only currently active downstream adapter units.
    The replacement arm periodically resets one mature least-useful unit.
    It does not reset the ELM, use phase boundaries, or replay examples.
    """
    def __init__(self, d, seed, width=96, k=24, replace=True, interval=48,
                 eta=.6, adapter_eta=.025, **unused):
        super().__init__(d, seed)
        self.width = int(width); self.k = min(int(k), self.width)
        self.replace = bool(replace); self.interval = max(1, int(interval))
        self.eta = float(eta); self.adapter_eta = float(adapter_eta)
        self.address = unit_rows(self.rng.normal(size=(self.width, d)))
        self.V = np.zeros((self.width, 4))
        self.utility = np.zeros(self.width)
        self.age = np.zeros(self.width, dtype=np.int64)
        self.usage = np.zeros(self.width, dtype=np.int64)
        self.replacements = 0

    def _encode(self, H):
        H, scalar = _structural_rows(H)
        positive = np.maximum(unit_rows(H)@self.address.T, 0)
        index = np.argsort(positive, axis=1, kind='stable')[:, -self.k:]
        mask = np.zeros_like(positive, dtype=bool)
        np.put_along_axis(mask, index, True, axis=1)
        masked = positive*mask
        return unit_rows(masked), masked, scalar

    def output(self, H):
        code, _, scalar = self._encode(H)
        result = code@self.V
        return result[0] if scalar else result

    def observe(self, h, label):
        h = unit_rows(np.asarray(h, dtype=np.float64))
        code, masked, _ = self._encode(h)
        u = code[0]; raw = masked[0]; active = raw > 0
        error = target4(label)-u@self.V
        direction = self.V@error
        # Derivative through normalization; hard Top-K membership is fixed
        # for this local step, with no update to inactive address vectors.
        grad = (direction-u*float(u@direction))/max(float(np.linalg.norm(raw)), 1e-8)
        grad *= active
        change = delta_change(u, error, self.eta)
        self.V += change
        if np.any(active):
            move = bounded_change(self.adapter_eta*np.outer(grad[active], h), .1)
            self.address[active] = unit_rows(self.address[active]+move)
        self.utility *= .99
        self.utility += .01*np.abs(u)*np.linalg.norm(self.V, axis=1)
        self.age += 1; self.usage[active] += 1; self.t += 1
        replaced = 0
        if self.replace and self.t % self.interval == 0:
            eligible = np.flatnonzero(self.age >= self.interval)
            if len(eligible):
                j = int(eligible[np.argmin(self.utility[eligible])])
                self.address[j] = unit_rows(self.rng.normal(size=self.d))
                self.V[j] = 0; self.utility[j] = 0; self.age[j] = 0
                self.usage[j] = 0; self.replacements += 1; replaced = 1
        return {'extra_labeled_presentations': 0,
                'update_norm': float(np.linalg.norm(change)),
                'replaced_units': replaced}

    def diagnostics(self):
        return {'allocated_units': self.width, 'replacements': self.replacements,
                'currently_unused_units': int(np.sum(self.usage == 0)),
                'mean_utility': float(self.utility.mean()),
                'minimum_unit_age': int(self.age.min())}


def build_structural_memory(spec, d, seed, route=None):
    kind, params = _structural_spec(spec)
    if kind == 'sparse_specialization':
        return SparseSpecializationMemory(d, seed, **params)
    if kind == 'allocation':
        return BoundedAllocationMemory(d, seed, **params)
    if kind == 'quantized_memory':
        params['quantized'] = True
        return BoundedAllocationMemory(d, seed, **params)
    if kind == 'silent_memory':
        return ContentBankMemory(d, seed, **params)
    if kind == 'context_gating':
        params.setdefault('recent_bias', 0.0)
        return ContentBankMemory(d, seed, **params)
    if kind == 'recycling':
        return RecyclingAdapterMemory(d, seed, **params)
    return None


def structural_mechanism_tests():
    """Bounded-capacity, inactive-unit, causal-revision and pure-access gates."""
    rng = np.random.default_rng(8151); d = 16
    H = unit_rows(rng.normal(size=(14, d)))
    specs = [
        {'kind': 'sparse_specialization', 'params': {'width': 24, 'k': 4, 'adaptive': a}}
        for a in (False, True)]
    specs += [{'kind': 'allocation', 'params': {'slots': 4, 'recycle': r, 'threshold': .999}}
              for r in (False, True)]
    specs += [{'kind': 'quantized_memory', 'params': {'slots': 4, 'code_k': 4,
                                                  'threshold': .999}}]
    specs += [{'kind': kind, 'params': {'banks': 3, 'routing': routing}}
              for kind in ('silent_memory', 'context_gating')
              for routing in ('inferred', 'fixed')]
    specs += [{'kind': 'recycling', 'params': {'width': 24, 'k': 4, 'interval': 4,
                                           'replace': replacement}}
              for replacement in (False, True)]
    for spec in specs:
        model = build_structural_memory(spec, d, 421)
        shapes = {k: v.shape for k, v in model.mutable_arrays().items()}
        byte_bound = model.mutable_bytes()
        for i, h in enumerate(H):
            info = model.observe(h, i % 4)
            assert info['extra_labeled_presentations'] == 0
            assert model.t == i+1
            assert np.all(np.isfinite(model.output(H)))
        assert model.mutable_bytes() == byte_bound
        assert {k: v.shape for k, v in model.mutable_arrays().items()} == shapes
        digest = model.state_digest()
        ordinary = model.output(H)
        recovered = model.reminded_output(H[3:], H[:3])
        assert recovered.shape == (len(H)-3, 4)
        assert np.all(np.isfinite(recovered))
        assert digest == model.state_digest(), 'Evaluation mutated learner state'
        assert np.array_equal(ordinary, model.output(H))
        if spec['kind'] in ('allocation', 'quantized_memory'):
            assert model.used <= model.slots
            assert ((model.replacements > 0) if model.recycle else (model.rejected > 0))
        if spec['kind'] == 'recycling':
            assert (model.replacements > 0) == model.replace
    # A conflicting observation revises a matching content slot, without IDs.
    allocation = BoundedAllocationMemory(d, 3, slots=2, threshold=.999)
    allocation.observe(H[0], 0); allocation.observe(H[0], 2)
    assert allocation.used == 1
    assert np.argmax(allocation.output(H[0])) == 2
    # Sparse local adaptation never moves losing addresses or outgoing rows.
    sparse = SparseSpecializationMemory(d, 3, width=24, k=4, adaptive=True)
    code, _, mask, _ = sparse._encode(H[0]); active = mask[0] & (code[0] > 0)
    before_address = sparse.address.copy(); before_V = sparse.V.copy()
    sparse.observe(H[0], 1)
    assert np.array_equal(sparse.address[~active], before_address[~active])
    assert np.array_equal(sparse.V[~active], before_V[~active])
    # Reminder access can differ while storage, centroids, counters and RNG do not.
    bank = ContentBankMemory(d, 7, banks=2, novelty=.999)
    bank.observe(H[0], 0); bank.observe(H[1], 1)
    state = bank.state_digest()
    assert not np.allclose(bank.reminded_output(H[2:], H[:1]),
                           bank.reminded_output(H[2:], H[1:2]))
    assert state == bank.state_digest()
    return {'structural_invariants': 'passed', 'structural_test_variants': len(specs)}


# Replay and content retrieval use only feature/label pairs actually presented.
# These are bounded engineering variants, not claimed paper reproductions.

class ObservedPairStore(MemoryBase):
    """Bounded observed-pair store; exact retained inputs refresh their own label.

    'reservoir' uses admission probabilities over external presentations, with
    retained exact duplicates coalesced by refresh. It is therefore explicitly a
    reservoir-with-refresh, not an exactly uniform reservoir over unique cues.
    'recent' replaces the least recently observed retained numerical input.
    No sample/cue identity, phase, future target, or evaluator object is stored.
    """
    def __init__(self, d, seed, capacity=8, policy='reservoir'):
        super().__init__(d, seed)
        if int(capacity) < 1 or policy not in ('reservoir', 'recent'):
            raise ValueError('Positive capacity and reservoir/recent policy required')
        self.capacity = int(capacity); self.policy = policy
        self.Hbuf = np.zeros((self.capacity, self.d), dtype=np.float64)
        self.Lbuf = np.zeros(self.capacity, dtype=np.int64)
        self.Sbuf = np.full(self.capacity, -1, dtype=np.int64)
        self.n = 0; self.seen = 0; self.refreshes = 0
        self.admissions = 0; self.replacements = 0

    def _insert_observed(self, h, label):
        h = np.asarray(h, dtype=np.float64)
        if h.shape != (self.d,):
            raise ValueError('observe expects one feature vector')
        if int(label) not in range(4):
            raise ValueError('Expected one of four observed labels')
        self.seen += 1
        matches = np.flatnonzero(np.all(self.Hbuf[:self.n] == h, axis=1))
        if len(matches):
            slot = int(matches[0]); self.refreshes += 1
        elif self.n < self.capacity:
            slot = self.n; self.n += 1; self.admissions += 1
        else:
            if self.policy == 'recent':
                slot = int(np.argmin(self.Sbuf[:self.n]))
            else:
                slot = int(self.rng.integers(self.seen))
                if slot >= self.capacity:
                    return
            self.admissions += 1; self.replacements += 1
        self.Hbuf[slot] = h; self.Lbuf[slot] = int(label)
        self.Sbuf[slot] = self.seen

    def _draw_preinsertion(self):
        if not self.n:
            return None
        slot = int(self.rng.integers(self.n))
        return self.Hbuf[slot].copy(), int(self.Lbuf[slot])

    def _buffer_diagnostics(self):
        return {'buffer_count': self.n, 'buffer_capacity': self.capacity,
                'observed_pair_presentations': self.seen,
                'exact_input_refreshes': self.refreshes,
                'buffer_admissions': self.admissions,
                'buffer_replacements': self.replacements}


class ReplayMemory(ObservedPairStore):
    def __init__(self, d, seed, capacity=8, policy='reservoir', eta=.6):
        super().__init__(d, seed, capacity, policy)
        self.V = np.zeros((d, 4), dtype=np.float64)
        self.eta = float(eta); self.extra_presentations = 0
        self.replayed_current_conflicts = 0

    def output(self, H):
        return np.asarray(H, dtype=np.float64) @ self.V

    def _plasticity(self, h, label):
        change = delta_change(h, target4(label)-self.output(h), self.eta)
        self.V += change
        return change

    def observe(self, h, label):
        h = np.asarray(h, dtype=np.float64)
        # Draw from the state before any insertion or correction of buffer labels.
        pair = self._draw_preinsertion()
        total_change = self._plasticity(h, label)
        if pair is not None:
            hp, yp = pair
            self.replayed_current_conflicts += int(np.array_equal(hp, h) and yp != int(label))
            total_change += self._plasticity(hp, yp)
            self.extra_presentations += 1
        self._insert_observed(h, label)
        self.t += 1
        return {'extra_labeled_presentations': int(pair is not None),
                'update_norm': float(np.linalg.norm(total_change))}

    def diagnostics(self):
        return dict(self._buffer_diagnostics(), replay_presentations=self.extra_presentations,
                    replayed_current_conflicts=self.replayed_current_conflicts)


class RepeatCurrentMemory(PlainMemory):
    """Exposure control: a second current-example update after the first event."""
    def __init__(self, d, seed, eta=.6):
        super().__init__(d, seed, eta)
        self.extra_presentations = 0

    def observe(self, h, label):
        h = np.asarray(h, dtype=np.float64)
        extra = int(self.t > 0)
        change = delta_change(h, target4(label)-self.output(h), self.eta)
        self.V += change
        if extra:
            second = delta_change(h, target4(label)-self.output(h), self.eta)
            self.V += second; change += second; self.extra_presentations += 1
        self.t += 1
        return {'extra_labeled_presentations': extra,
                'update_norm': float(np.linalg.norm(change))}

    def diagnostics(self):
        return {'repeated_current_presentations': self.extra_presentations}


class RetrievalMemory(ObservedPairStore):
    """Ordinary learned weights plus bounded content-addressed observed targets.

    Hard: nearest stored feature if cosine >= .90. Soft: normalized exp(cos/.10)
    over stored features with cosine >= .75. Either mixes .75 retrieved target
    and .25 parametric output; when no match qualifies, use parametric output.
    Thresholds are fixed before evaluation and are included in diagnostics.
    Prediction is pure: lookup never refreshes slots, mutates counts, or learns.
    """
    def __init__(self, d, seed, capacity=8, mode='hard', policy='recent', eta=.6,
                 hard_threshold=.90, soft_threshold=.75, temperature=.10, mix=.75):
        super().__init__(d, seed, capacity, policy)
        if mode not in ('hard', 'soft'):
            raise ValueError('Retrieval mode must be hard or soft')
        self.mode = mode; self.eta = float(eta)
        self.hard_threshold = float(hard_threshold); self.soft_threshold = float(soft_threshold)
        self.temperature = float(temperature); self.mix = float(mix)
        if self.temperature <= 0 or not 0 <= self.mix <= 1:
            raise ValueError('Invalid retrieval temperature/mix')
        self.V = np.zeros((d, 4), dtype=np.float64)
        self.online_lookup_hits = 0; self.online_lookup_similarity_sum = 0.

    def _lookup(self, H):
        H = np.asarray(H, dtype=np.float64)
        matrix = np.atleast_2d(H)
        out = np.zeros((len(matrix), 4), dtype=np.float64)
        hits = np.zeros(len(matrix), dtype=bool)
        best = np.zeros(len(matrix), dtype=np.float64)
        if not self.n:
            return out, hits, best
        similarity = unit_rows(matrix) @ unit_rows(self.Hbuf[:self.n]).T
        # Protect fixed cosine thresholds against floating-point overshoot.
        similarity = np.clip(similarity, -1., 1.)
        best = similarity.max(axis=1)
        targets = np.eye(4, dtype=np.float64)[self.Lbuf[:self.n]]
        if self.mode == 'hard':
            idx = similarity.argmax(axis=1)
            hits = best >= self.hard_threshold
            out = targets[idx]
        else:
            eligible = similarity >= self.soft_threshold
            hits = eligible.any(axis=1)
            weights = np.exp((similarity-best[:, None])/self.temperature) * eligible
            weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
            out = weights @ targets
        return out, hits, best

    def output(self, H):
        H = np.asarray(H, dtype=np.float64)
        base = np.atleast_2d(H) @ self.V
        retrieved, hits, _ = self._lookup(H)
        alpha = self.mix * hits[:, None]
        scores = (1.-alpha)*base + alpha*retrieved
        return scores[0] if H.ndim == 1 else scores

    def observe(self, h, label):
        h = np.asarray(h, dtype=np.float64)
        # Parametric updates stay identical to ordinary delta learning. Retrieval
        # changes inference, not which residual is taught into the fallback head.
        _, hit, best = self._lookup(h)
        self.online_lookup_hits += int(hit[0])
        self.online_lookup_similarity_sum += float(best[0])
        change = delta_change(h, target4(label)-h@self.V, self.eta)
        self.V += change
        self._insert_observed(h, label); self.t += 1
        return {'extra_labeled_presentations': 0,
                'update_norm': float(np.linalg.norm(change))}

    def diagnostics(self):
        return dict(self._buffer_diagnostics(),
                    online_lookup_hit_fraction=self.online_lookup_hits/max(self.t, 1),
                    online_lookup_mean_similarity=self.online_lookup_similarity_sum/max(self.t, 1),
                    hard_cosine_threshold=self.hard_threshold,
                    soft_cosine_threshold=self.soft_threshold,
                    retrieval_temperature=self.temperature, retrieval_mix=self.mix)


class ReplayConsolidationMemory(ObservedPairStore):
    """The exact ConsolidationMemory core plus one preinsertion replay update.

    Both plastic updates precede the single external-event consolidation tick.
    Extra supervised presentations are counted without speeding slow-state time.
    """
    def __init__(self, d, seed, capacity=8, policy='reservoir', states=2,
                 feedback=False, eta=.6, coupling=.04):
        super().__init__(d, seed, capacity, policy)
        self.core = ConsolidationMemory(d, seed, states=states, feedback=feedback,
                                        eta=eta, coupling=coupling)
        self.extra_presentations = 0; self.replayed_current_conflicts = 0

    def output(self, H):
        return self.core.output(H)

    def observe(self, h, label):
        h = np.asarray(h, dtype=np.float64)
        pair = self._draw_preinsertion()
        if pair is not None:
            self.replayed_current_conflicts += int(np.array_equal(pair[0], h) and pair[1] != int(label))
        result = dict(self.core.observe_with_replay(h, label, pair))
        self.extra_presentations += int(pair is not None)
        self._insert_observed(h, label); self.t += 1
        result['extra_labeled_presentations'] = int(pair is not None)
        return result

    def mutable_arrays(self):
        own = super().mutable_arrays()
        own.update({'core.'+k: v for k, v in self.core.mutable_arrays().items()})
        return own

    def state_digest(self):
        return hashlib.sha256((super().state_digest()+self.core.state_digest()).encode()).hexdigest()

    def diagnostics(self):
        result = dict(self._buffer_diagnostics(), replay_presentations=self.extra_presentations,
                      replayed_current_conflicts=self.replayed_current_conflicts,
                      consolidation_external_ticks=self.core.t)
        result.update({'core_'+k: v for k, v in self.core.diagnostics().items()})
        return result


def build_replay_memory(spec, d, seed, route=None):
    p = dict(spec.get('params', {})); p.update({k: v for k, v in spec.items() if k != 'params'})
    kind = p.get('kind'); eta = float(p.get('eta', .6))
    if kind == 'repeat_current':
        return RepeatCurrentMemory(d, seed, eta)
    if kind == 'replay':
        return ReplayMemory(d, seed, int(p.get('capacity', 8)), p.get('policy', 'reservoir'), eta)
    if kind == 'retrieval':
        return RetrievalMemory(d, seed, int(p.get('capacity', 8)), p.get('mode', 'hard'),
                               p.get('policy', 'recent'), eta,
                               p.get('hard_threshold', .90), p.get('soft_threshold', .75),
                               p.get('temperature', .10), p.get('mix', .75))
    if kind == 'replay_consolidation':
        return ReplayConsolidationMemory(d, seed, int(p.get('capacity', 8)),
                                         p.get('policy', 'reservoir'), int(p.get('states', 2)),
                                         bool(p.get('feedback', False)), eta,
                                         float(p.get('coupling', .04)))
    return None


def replay_mechanism_tests():
    """Small deterministic semantics checks, independent of evaluation tasks."""
    report = {}; dim = 6
    eye = np.eye(dim)
    # A huge replay learning rate would not matter: first event cannot replay the
    # freshly seen item. Its weights must equal the ordinary single update.
    replay = ReplayMemory(dim, 17, capacity=3, policy='recent')
    plain = PlainMemory(dim, 17)
    first = replay.observe(eye[0], 1); plain.observe(eye[0], 1)
    assert first['extra_labeled_presentations'] == 0
    assert np.array_equal(replay.V, plain.V)
    assert replay.n == 1 and replay.Lbuf[0] == 1
    # A second occurrence with a changed target replays the old observed pair
    # and only then refreshes that exact stored vector's target.
    expected = plain.V.copy()
    expected += delta_change(eye[0], target4(2)-eye[0]@expected)
    expected += delta_change(eye[0], target4(1)-eye[0]@expected)
    second = replay.observe(eye[0], 2)
    assert np.allclose(replay.V, expected, atol=1e-14)
    assert replay.n == 1 and replay.Lbuf[0] == 2
    assert replay.replayed_current_conflicts == 1
    assert second['extra_labeled_presentations'] == 1 and replay.t == 2
    report['preinsertion_replay_and_exact_refresh'] = True

    for policy in ('recent', 'reservoir'):
        model = ReplayMemory(dim, 22, capacity=3, policy=policy)
        history = []
        for i in range(27):
            h = eye[i % dim].copy(); label = i % 4
            history.append((h.copy(), label)); model.observe(h, label)
            assert model.n <= model.capacity
            for j in range(model.n):
                assert any(np.array_equal(hh, model.Hbuf[j]) and yy == model.Lbuf[j]
                           for hh, yy in history)
        digest = model.state_digest()
        model.output(eye); model.output(eye[0])
        assert digest == model.state_digest()
        assert model.t == 27 and model.extra_presentations == 26
    report['seen_only_bounded_buffer_and_pure_prediction'] = True

    hard = RetrievalMemory(dim, 32, 3, 'hard')
    soft = RetrievalMemory(dim, 32, 3, 'soft')
    plain = PlainMemory(dim, 32)
    for i in range(15):
        h = eye[i % dim]; label = i % 4
        hard.observe(h, label); soft.observe(h, label); plain.observe(h, label)
        assert np.array_equal(hard.Hbuf, soft.Hbuf)
        assert np.array_equal(hard.Lbuf, soft.Lbuf)
        assert np.array_equal(hard.Sbuf, soft.Sbuf)
        assert np.array_equal(hard.V, plain.V) and np.array_equal(soft.V, plain.V)
    for model in (hard, soft):
        digest = model.state_digest(); matrix = model.output(eye)
        assert matrix.shape == (dim, 4) and model.output(eye[0]).shape == (4,)
        assert np.array_equal(matrix[0], model.output(eye[0]))
        assert digest == model.state_digest()
    report['retrieval_shared_population_and_ordinary_fallback_updates'] = True

    repeat = RepeatCurrentMemory(dim, 5)
    replay = ReplayMemory(dim, 5)
    for i in range(7):
        x = eye[i % dim]; y = i % 4
        r1 = repeat.observe(x, y); r2 = replay.observe(x, y)
        assert r1['extra_labeled_presentations'] == r2['extra_labeled_presentations']
    assert repeat.t == replay.t == 7
    report['repeat_current_exposure_schedule'] = True

    # The consolidated variant is tested when its class is available after
    # assembly; no substitute model is silently used when it is absent.
    if 'ConsolidationMemory' in globals():
        core = ConsolidationMemory(dim, 11)
        hybrid = ReplayConsolidationMemory(dim, 11, capacity=3)
        r0 = core.observe(eye[0], 1); r1 = hybrid.observe(eye[0], 1)
        assert r1['extra_labeled_presentations'] == 0
        assert core.t == hybrid.core.t == hybrid.t == 1
        for name, arr in core.mutable_arrays().items():
            assert np.array_equal(arr, hybrid.core.mutable_arrays()[name])
        for i in range(1, 8):
            hybrid.observe(eye[i % dim], i % 4)
            assert hybrid.t == hybrid.core.t == i+1
        assert hybrid.extra_presentations == 7
        digest = hybrid.state_digest(); hybrid.output(eye)
        assert digest == hybrid.state_digest()
        assert hybrid.mutable_bytes() >= hybrid.core.mutable_bytes()+hybrid.Hbuf.nbytes
        report['replay_consolidation_single_clock_and_first_event_equivalence'] = True
    return report


# V23 consolidated runner. Definitions above include the V22 encoder and anatomy.
SOURCE_V22_SHA256 = '2faf4f635a27c503042e4a8a6eac952ea1045e18706548700d06836eaf6ca947'
V23_VERSION = '23.0.0'


class LegacyMemory(MemoryBase):
    def __init__(self, d, seed, route, codes, cfg, kind, eta=None):
        super().__init__(d, seed)
        if kind == 'cumulative_ridge':
            self.inner = CumulativeRidge(d, codes, cfg.ridge)
        else:
            local_cfg = argparse.Namespace(**vars(cfg))
            if eta is not None:
                local_cfg.learning_rate = eta
            self.inner = LocalLearner(kind, route, codes, local_cfg)
        self.kind = kind

    def output(self, H):
        return self.inner.output(H)

    def observe(self, h, label):
        _, info = self.inner.observe(h, label)
        self.t += 1
        return {**info, 'extra_labeled_presentations': 0}

    def mutable_arrays(self):
        ans = {'V': self.inner.V}
        if hasattr(self.inner, 'inverse'):
            ans['inverse'] = self.inner.inverse
        return ans


def arm_manifest():
    specs = []
    def add(name, family, kind, params=None, feature='sparse', note=''):
        specs.append(dict(name=name, family=family, kind=kind, params=params or {},
                          feature=feature, note=note))
    add('ordinary', 'reference', 'legacy', {'mode': 'uniform_delta'})
    add('ordinary_slow', 'reference', 'legacy', {'mode': 'uniform_delta', 'eta': .15},
        note='Low-rate retention/plasticity control.')
    add('v21_input', 'reference', 'legacy', {'mode': 'g1_feedback'})
    add('v22_joint', 'reference', 'legacy', {'mode': 'joint_feedback'})
    add('v22_scrambled', 'reference', 'legacy', {'mode': 'joint_scrambled'})
    add('historical_ridge', 'reference', 'legacy', {'mode': 'cumulative_ridge'},
        note='Covariance memory and equally weighted obsolete labels are explicit costs/limitations.')
    for name, params in [
        ('pair_importance', {'granularity': 'pair', 'strength': 3}),
        ('pair_importance_strong', {'granularity': 'pair', 'strength': 12}),
        ('neuron_availability', {'granularity': 'neuron', 'strength': 3}),
        ('pair_importance_shuffled', {'granularity': 'pair', 'strength': 3, 'scramble': True}),
        ('joint_plus_importance', {'granularity': 'pair', 'strength': 3, 'anatomical': True}),
        ('importance_surprise_release', {'granularity': 'pair', 'strength': 3, 'release': True})]:
        add(name, 'importance_and_availability', 'importance', params)
    for name, noise in [('bayes_stable', 0.), ('bayes_reopening', .003)]:
        add(name, 'uncertainty', 'bayes', {'process_noise': noise},
            note='Diagonal Bayesian filter proxy; not exact MESU.')
    for name, rank, leak in [('projection16_strict',16,0.), ('projection48_strict',48,0.),
                            ('projection16_leaky',16,.15)]:
        add(name, 'subspace', 'projection', {'rank': rank, 'leak': leak},
            note='Causal incremental input basis; no task-boundary SVD or oracle past probes.')
    for name, states, feedback in [('fast_slow2',2,False),('fast_slow3',3,False),
                                    ('feedback_consolidation',2,True)]:
        add(name, 'consolidation', 'consolidation', {'states': states, 'feedback': feedback})
    for levels in [4,8]:
        add('cascade'+str(levels), 'discrete_metaplasticity', 'cascade', {'levels': levels},
            note='Bounded stochastic depth proxy, not exact Fusi cascade.')
    add('active_forgetting', 'controlled_forgetting', 'active_forgetting',
        {'fast_decay': .01, 'slow_decay': .0001}, note='Task-free ensemble proxy, not published CAF protocol.')
    for mode in ['immediate','smoothed']:
        add('surprise_'+mode, 'surprise', 'surprise', {'mode': mode})
    for mode in ['hebbian','residual']:
        add('fastweights_'+mode, 'fast_association', 'fastweights', {'mode': mode})
    for width, k in [(96,8),(384,24)]:
        for adaptive in [False,True]:
            add('sparse%d_%s'%(width,'adaptive' if adaptive else 'fixed'), 'sparse_geometry',
                'sparse_specialization', {'width': width, 'k': k, 'adaptive': adaptive}, 'dense',
                'Positive normalized competitive address proxy; additional width is counted.')
    for name, slots, recycle, quantized in [('allocate32',32,True,False),
           ('allocate128',128,True,False), ('allocate32_no_recycle',32,False,False),
           ('quantized_allocate128',128,True,True)]:
        add(name, 'bounded_allocation', 'allocation',
            {'slots': slots, 'threshold': .9, 'recycle': recycle, 'quantized': quantized},
            note='ART/SQHN-inspired bounded slot proxy; not hierarchical MAP SQHN or unbounded expansion.')
    for routing in ['inferred','fixed']:
        add('silent_'+routing, 'storage_and_access', 'silent_memory', {'banks': 8, 'routing': routing})
        add('context_'+routing, 'context_routing', 'context_gating', {'banks': 4, 'routing': routing},
            note='Routes inferred from numerical input; no externally supplied task or phase ID.')
    for replace in [False,True]:
        add('adapter_'+('recycled' if replace else 'retained'), 'capacity_recycling', 'recycling',
            {'width':96,'k':24,'replace':replace,'interval':48}, 'dense',
            'Trainable downstream address units; the shared ELM itself stays frozen.')
    for policy in ['reservoir','recent']:
        for capacity in [8,32]:
            add('replay_%s%d'%(policy,capacity), 'replay_reference', 'replay',
                {'policy':policy,'capacity':capacity})
    add('repeat_current', 'exposure_control', 'repeat_current')
    for capacity in [8,32]:
        for mode in ['hard','soft']:
            add('retrieve_%s%d'%(mode,capacity), 'episodic_retrieval', 'retrieval',
                {'mode':mode,'policy':'recent','capacity':capacity})
    for capacity in [8,32]:
        add('fastslow_replay%d'%capacity, 'combination', 'replay_consolidation',
            {'policy':'reservoir','capacity':capacity,'states':2,'feedback':False})
    assert len({s['name'] for s in specs}) == len(specs)
    return specs


def make_memory(spec, seed, route, codes, cfg):
    # Family RNGs are shared across variants so comparison differences do not
    # begin with an unrelated random address map or buffer draw stream.
    family_seed = seed_for(seed, 'v23_family_'+spec['family'])
    if spec['kind'] == 'legacy':
        return LegacyMemory(96, family_seed, route, codes, cfg,
                            spec['params']['mode'], spec['params'].get('eta'))
    for factory in [build_weight_memory, build_structural_memory, build_replay_memory]:
        model = factory(spec, 96, family_seed, route)
        if model is not None:
            return model
    raise ValueError('Unimplemented arm '+spec['name'])


def build_stream(cfg, seed):
    """Environment-only data and labels; none of its IDs enter a learner."""
    rg = rng_for(seed, 'stream_data')
    n0 = 2*cfg.group_cues
    bases = cue_bases(rg, n0)
    train0 = render_cues(bases, cfg.variants, rg, cfg.noise)
    probe0 = render_cues(bases, cfg.probe_variants,
                        rng_for(seed,'heldout_probe_data'),cfg.noise)
    original0 = np.tile(np.arange(4),math.ceil(n0/4))[:n0]
    for group in (slice(0,cfg.group_cues),slice(cfg.group_cues,n0)):
        original0[group] = rg.permutation(original0[group])
    reverse_ids = np.r_[rg.choice(cfg.group_cues,cfg.reverse_cues//2,False),
                       cfg.group_cues+rg.choice(cfg.group_cues,cfg.reverse_cues//2,False)]
    trains, probes, labels = [train0], [probe0], [original0]
    # Separate RNG namespaces preserve every original V22 draw exactly.
    for wave in range(1+cfg.stress_waves):
        r = rng_for(seed,'v23_new_group_%d'%wave)
        b = cue_bases(r,cfg.group_cues)
        trains.append(render_cues(b,cfg.variants,r,cfg.noise))
        probes.append(render_cues(b,cfg.probe_variants,
                                 rng_for(seed,'v23_probe_group_%d'%wave),cfg.noise))
        lab = np.tile(np.arange(4),math.ceil(cfg.group_cues/4))[:cfg.group_cues]
        labels.append(r.permutation(lab))
    train, probe, original = np.concatenate(trains), np.concatenate(probes), np.concatenate(labels)
    N = len(original); current = original.copy()
    schedule = []; order = rng_for(seed,'stream_order')
    stream_hash = hashlib.sha256(); external = 0
    seen = np.zeros(N,bool); ever_revised = np.zeros(N,bool)
    initial_reverse = np.zeros(N,bool); initial_reverse[reverse_ids] = True
    def phase(name, ids, is_revision=False, revise_offset=None, use_old_order=False):
        nonlocal external
        ids = np.asarray(ids,dtype=int)
        if is_revision:
            if revise_offset is None:
                current[ids] = (current[ids]+1)%4
            else:
                current[ids] = (original[ids]+revise_offset)%4
            ever_revised[ids] = True
        phase_order = order if use_old_order else rng_for(seed,'v23_order_'+name)
        checkpoints = []
        for p in range(cfg.passes):
            pairs = [(int(i),v) for i in ids for v in range(cfg.variants)]
            events = [(pairs[j][0],pairs[j][1],int(current[pairs[j][0]]))
                      for j in phase_order.permutation(len(pairs))]
            for i,v,y in events:
                if name in ('acquire_A','interfere_B','revise_subset'):
                    stream_hash.update(np.asarray([i,v,y],dtype='<i8').tobytes())
            external += len(events); seen[ids] = True
            checkpoints.append(dict(pass_index=p+1,events=events,external=external))
        schedule.append(dict(name=name,ids=ids.tolist(),revision=is_revision,
            targets=current.copy(),seen=seen.copy(),ever_revised=ever_revised.copy(),
            checkpoints=checkpoints))
    phase('acquire_A',np.arange(cfg.group_cues),use_old_order=True)
    phase('interfere_B',np.arange(cfg.group_cues,n0),use_old_order=True)
    phase('revise_subset',reverse_ids,True,1,True)
    prefix_hash = stream_hash.hexdigest()
    phase('interfere_C',np.arange(n0,n0+cfg.group_cues))
    phase('revise_again',reverse_ids,True,2)
    for wave in range(cfg.stress_waves):
        end = n0+(wave+2)*cfg.group_cues
        phase('novel_%02d'%(wave+1),np.arange(end-cfg.group_cues,end))
        # Revision cohorts selected in the environment; no identity or novelty
        # marker is supplied to memory. Alternate old and mixed cohorts.
        eligible = np.arange(n0 if wave%2==0 else end)
        r = rng_for(seed,'v23_stress_revision_%d'%wave)
        ids = r.choice(eligible,cfg.reverse_cues,replace=False)
        phase('revision_%02d'%(wave+1),ids,True)
    prefix_costs = dict(seed=seed,stream_sha256=prefix_hash,
        online_unique_training_inputs_sha256=sha_bytes(train0.tobytes()),
        heldout_probe_inputs_sha256=sha_bytes(probe0.tobytes()))
    prefix_check = check_v20_stream_compatibility(cfg,[prefix_costs])[0]
    full_hash = hashlib.sha256()
    pair_set = set()
    for ph in schedule:
        for cp in ph['checkpoints']:
            for i,v,y in cp['events']:
                full_hash.update(np.asarray([i,v,y],dtype='<i8').tobytes())
                pair_set.add((i,v,y))
    return dict(train=train,probe=probe,original=original,schedule=schedule,
        initial_reverse=initial_reverse,prefix_check=prefix_check,prefix_costs=prefix_costs,
        unique_identities=N,unique_inputs=N*cfg.variants,
        distinct_input_outcome_pairs=len(pair_set),external=external,
        full_stream_sha256=full_hash.hexdigest())


def prepare_features(cfg, seed, stream):
    model, mean, sd, development = develop_encoder(cfg,seed)
    feats = {}
    for key in ['train','probe']:
        x = stream[key]; shape = x.shape[:2]
        # Encode original A/B separately to retain the original final batch.
        cut = 2*cfg.group_cues
        chunks = [encode(model,x[:cut].reshape(-1,12,32)),
                  encode(model,x[cut:].reshape(-1,12,32))]
        raw = np.concatenate(chunks)
        dense = unit_rows(np.clip((raw-mean)/sd,-5,5).astype(np.float64))
        sparse = sparse_features(raw,mean,sd,cfg.active_fraction)
        feats[key] = dict(dense=dense.reshape(*shape,96),sparse=sparse.reshape(*shape,96))
    final_hash = sha_bytes(b''.join(t.detach().cpu().numpy().tobytes() for t in model.state_dict().values()))
    if final_hash != development['encoder_sha256']:
        raise AssertionError('Encoder changed during feature preparation.')
    development['online_encoder_updates'] = 0
    development['cached_feature_bytes'] = sum(a.nbytes for v in feats.values() for a in v.values())
    development['unique_encoder_cache_forwards'] = len(stream['train'])*(cfg.variants+cfg.probe_variants)
    return feats, development


def accuracy(scores, targets, mask, variants):
    mask = np.asarray(mask,dtype=bool)
    if not mask.any():
        return None
    z = np.asarray(scores).reshape(len(targets),variants,4)
    return float(np.mean(z[mask].argmax(-1)==np.asarray(targets)[mask,None]))


def checkpoint_metrics(model, H, ph, stream, cfg):
    before = model.state_digest()
    z = np.asarray(model.output(H.reshape(-1,96)))
    if z.shape != (len(H)*cfg.probe_variants,4) or not np.isfinite(z).all():
        raise FloatingPointError('Nonfinite or incorrectly shaped evaluation scores.')
    if model.state_digest()!=before:
        raise AssertionError('Prediction mutated online memory or its RNG.')
    N=len(H); seen=ph['seen']; ids=np.zeros(N,bool);ids[ph['ids']]=True
    a=np.arange(N)<cfg.group_cues
    b=(np.arange(N)>=cfg.group_cues)&(np.arange(N)<2*cfg.group_cues)
    c=(np.arange(N)>=2*cfg.group_cues)&(np.arange(N)<3*cfg.group_cues)
    masks=dict(all_current=seen,A_current=a&seen,B_current=b&seen,C_current=c&seen,
        initial_revised=stream['initial_reverse']&seen,
        original_never_revised=seen&~ph['ever_revised'],
        phase_current=ids&seen,prior_current=seen&~ids,
        initial_unchanged=(a|b)&~stream['initial_reverse']&seen)
    out={k:accuracy(z,ph['targets'],m,cfg.probe_variants) for k,m in masks.items()}
    flat = z.reshape(N,cfg.probe_variants,4)[seen]
    target=np.eye(4)[ph['targets'][seen]][:,None,:]
    out['current_mse']=float(np.mean((flat-target)**2))
    out['seen_identities']=int(seen.sum())
    return out


def aggregate_info(acc, info):
    for k,v in (info or {}).items():
        if isinstance(v,(int,float,bool,np.integer,np.floating)) and np.isfinite(float(v)):
            item=acc.setdefault(k,[0.,0]);item[0]+=float(v);item[1]+=1


def reminder_diagnostic(model, spec, ph, stream, feat, cfg, seed):
    if spec['kind'] not in ('silent_memory','context_gating','retrieval'):
        return []
    a=np.flatnonzero(ph['seen'] & (np.arange(len(ph['seen']))<cfg.group_cues))
    other=np.flatnonzero(ph['seen'] & (np.arange(len(ph['seen']))>=cfg.group_cues))
    if len(a)<4 or not len(other):
        return []
    # Reminder identities and target identities are disjoint. Reminder inputs
    # are observed sensory variants with NO outcomes and no storage updates.
    count=min(3,len(a)//3,len(other))
    r=rng_for(seed,'v23_reminder_'+ph['name'])
    remids=r.choice(a,count,False); wrongids=r.choice(other,count,False)
    testids=np.setdiff1d(a,remids)
    H=feat['probe'][spec['feature']][testids].reshape(-1,96)
    digest=model.state_digest(); rows=[]
    base=model.output(H)
    for condition,ids in [('no_reminder',[]),('A_reminder',remids),('other_reminder',wrongids)]:
        if len(ids):
            cues=feat['train'][spec['feature']][ids,0]
            scores=model.reminded_output(H,cues)
        else:
            scores=base
        scores=np.asarray(scores)
        if scores.shape!=base.shape or not np.isfinite(scores).all():
            raise AssertionError('Invalid reminder score shape or value.')
        if model.state_digest()!=digest:
            raise AssertionError('Frozen-storage reminder changed memory, gate state, clock, or RNG.')
        pred=scores.reshape(len(testids),cfg.probe_variants,4).argmax(-1)
        curr=ph['targets'][testids]; old=stream['original'][testids]
        revised=curr!=old
        rows.append(dict(condition=condition,reminder_inputs=len(ids),
            scored_identities=len(testids),current_accuracy=float(np.mean(pred==curr[:,None])),
            original_label_accuracy=float(np.mean(pred==old[:,None])),
            obsolete_label_accuracy=float(np.mean(pred[revised]==old[revised,None])) if revised.any() else None,
            frozen_storage_verified=True,reminder_test_identity_overlap=0))
    return rows


def capacity_reference(ph, stream, feat, cfg):
    # Diagnostic only: choose all distinct observed numeric training inputs
    # once, with the environment's current label, including superseded labels.
    # This deliberately privileged reference is NEVER supplied to a learner.
    ids=np.flatnonzero(ph['seen'])
    H=feat['train']['sparse'][ids].reshape(-1,96)
    T=np.repeat(np.eye(4)[ph['targets'][ids]],cfg.variants,axis=0)
    V=np.linalg.solve(H.T@H+cfg.ridge*np.eye(96),H.T@T)
    P=feat['probe']['sparse'][ids].reshape(-1,96)
    s=np.linalg.svd(H,compute_uv=False)
    rank=int(np.sum(s>max(H.shape)*np.finfo(float).eps*s[0]))
    norms=unit_rows(H)
    G=norms@norms.T
    mask=~np.eye(len(H),dtype=bool)
    return dict(phase=ph['name'],unique_fit_inputs=len(H),rank=rank,
        mean_absolute_feature_overlap=float(np.mean(np.abs(G[mask]))) if mask.any() else 0.,
        heldout_current_accuracy=float(np.mean((P@V).argmax(-1)==np.repeat(ph['targets'][ids],cfg.probe_variants))),
        status='evaluator_only_current_label_feasibility_reference_not_online_not_an_upper_bound')


def run_experiment_seed(cfg, seed, anatomy, specs):
    started=time.perf_counter()
    stream=build_stream(cfg,seed)
    print('  developing shared encoder and caching %d identities'%stream['unique_identities'],flush=True)
    feat,development=prepare_features(cfg,seed,stream)
    codes=make_codebook(seed);route=Route(anatomy,seed,codes)
    models={s['name']:make_memory(s,seed,route,codes,cfg) for s in specs}
    metrics=[];diagnostics=[];reminders=[];damage=[];failures=[];capacity=[]
    extras={s['name']:0 for s in specs};wall={s['name']:0. for s in specs}
    peak={s['name']:models[s['name']].mutable_bytes() for s in specs}
    failed=set();actual_seen=np.zeros(stream['unique_identities'],bool);external=0
    for ph in stream['schedule']:
        phase_diag={s['name']:{} for s in specs}
        phase_damage={s['name']:dict(n=0,actual_drift=0.,ordinary_drift=0.,
                        old_mse_change=0.,ordinary_old_mse_change=0.,safer_writes=0) for s in specs}
        for cp in ph['checkpoints']:
            online_correct={s['name']:0 for s in specs}
            online_count={s['name']:0 for s in specs}
            for identity,variant,label in cp['events']:
                external+=1
                protected=actual_seen.copy();protected[identity]=False
                if ph['revision']:
                    protected[ph['ids']]=False
                candidate_ids=np.flatnonzero(protected)
                measure=(cfg.diagnostic_interval>0 and external%cfg.diagnostic_interval==0
                         and len(candidate_ids)>0)
                # Deterministic evaluator subsampling, inaccessible to learners.
                if measure:
                    offset=(external//max(1,cfg.diagnostic_interval))%len(candidate_ids)
                    selected=np.roll(candidate_ids,-offset)[:min(8,len(candidate_ids))]
                for spec in specs:
                    name=spec['name']
                    if name in failed:continue
                    model=models[name];h=feat['train'][spec['feature']][identity,variant]
                    tick=time.perf_counter()
                    try:
                        before=np.asarray(model.output(h))
                        if before.shape!=(4,) or not np.isfinite(before).all():
                            raise FloatingPointError('Invalid prequential scores.')
                        online_correct[name]+=int(np.argmax(before)==label)
                        online_count[name]+=1
                        do_damage=measure and spec['feature']=='sparse'
                        if do_damage:
                            H=feat['probe']['sparse'][selected].reshape(-1,96)
                            old_before=np.asarray(model.output(H))
                            old_target=np.repeat(np.eye(4)[ph['targets'][selected]],cfg.probe_variants,axis=0)
                            reference_shift=H@delta_change(h,target4(label)-before,cfg.learning_rate)
                        info=model.observe(h,label)
                        if model.t!=external:
                            raise AssertionError('Model external clock differs from environment: %d vs %d'%(model.t,external))
                        extra=float(info.get('extra_labeled_presentations',0))
                        if extra<0 or int(extra)!=extra:
                            raise AssertionError('Invalid presentation accounting.')
                        extras[name]+=int(extra);aggregate_info(phase_diag[name],info)
                        if do_damage:
                            old_after=np.asarray(model.output(H))
                            shift=old_after-old_before
                            if not np.isfinite(shift).all():raise FloatingPointError('Nonfinite post-write scores.')
                            a=float(np.mean(shift**2));b=float(np.mean(reference_shift**2))
                            d=phase_damage[name];d['n']+=1;d['actual_drift']+=a;d['ordinary_drift']+=b
                            d['old_mse_change']+=float(np.mean((old_after-old_target)**2-(old_before-old_target)**2))
                            d['ordinary_old_mse_change']+=float(np.mean((old_before+reference_shift-old_target)**2-(old_before-old_target)**2))
                            d['safer_writes']+=int(a<b)
                    except Exception as exc:
                        failed.add(name)
                        failures.append(dict(seed=seed,arm=name,phase=ph['name'],external=external,
                            exception=type(exc).__name__,message=str(exc)))
                        print('  FAILED %s at %d: %s'%(name,external,exc),flush=True)
                    wall[name]+=time.perf_counter()-tick
                actual_seen[identity]=True
            for spec in specs:
                name=spec['name']
                if name in failed:continue
                model=models[name]
                try:
                    m=checkpoint_metrics(model,feat['probe'][spec['feature']],ph,stream,cfg)
                    row=dict(seed=seed,arm=name,family=spec['family'],phase=ph['name'],
                        pass_index=cp['pass_index'],external_presentations=external,
                        extra_labeled_presentations=extras[name],total_labeled_presentations=external+extras[name],
                        prequential_accuracy=online_correct[name]/max(1,online_count[name]),**m)
                    metrics.append(row)
                    peak[name]=max(peak[name],model.mutable_bytes())
                except Exception as exc:
                    failed.add(name);failures.append(dict(seed=seed,arm=name,phase=ph['name'],external=external,
                        exception=type(exc).__name__,message=str(exc)))
        for spec in specs:
            name=spec['name']
            if name in failed:continue
            model=models[name];stats=model.diagnostics()
            stats={k:float(v) for k,v in stats.items()
                   if isinstance(v,(int,float,bool,np.integer,np.floating)) and np.isfinite(float(v))}
            avg={'write_mean_'+k:v[0]/v[1] for k,v in phase_diag[name].items() if v[1]}
            diagnostics.append(dict(seed=seed,arm=name,phase=ph['name'],external_presentations=external,
                                    mutable_numeric_bytes=model.mutable_bytes(),**stats,**avg))
            if ph['name'] in ('interfere_B','interfere_C','revise_again') or ph is stream['schedule'][-1]:
                for r in reminder_diagnostic(model,spec,ph,stream,feat,cfg,seed):
                    reminders.append(dict(seed=seed,arm=name,phase=ph['name'],**r))
            d=phase_damage[name]
            if d['n']:
                damage.append(dict(seed=seed,arm=name,phase=ph['name'],sampled_writes=d['n'],
                    displacement_ratio=d['actual_drift']/d['ordinary_drift'] if d['ordinary_drift']>1e-20 else None,
                    fraction_lower_displacement=d['safer_writes']/d['n'],
                    mean_old_mse_change=d['old_mse_change']/d['n'],
                    mean_ordinary_old_mse_change=d['ordinary_old_mse_change']/d['n'],
                    actual_drift_sum=d['actual_drift'],ordinary_drift_sum=d['ordinary_drift'],
                    comparison='same_expressed_state_additive_input_delta_reference; includes_all_actual_state_changes'))
        if ph['name'] in ('interfere_C','revise_again') or ph is stream['schedule'][-1]:
            capacity.append(capacity_reference(ph,stream,feat,cfg))
        print('  seed %d: %-16s %5d external presentations; %d/%d arms active'%
              (seed,ph['name'],external,len(specs)-len(failed),len(specs)),flush=True)
    costs=[]
    for spec in specs:
        name=spec['name'];model=models[name]
        fixed=float(model.diagnostics().get('fixed_bytes',0))
        costs.append(dict(seed=seed,arm=name,status='failed' if name in failed else 'completed',
            external_presentations_completed=model.t,extra_labeled_presentations=extras[name],
            total_labeled_presentations=model.t+extras[name],peak_mutable_numeric_bytes=peak[name],
            reported_fixed_numeric_bytes=int(fixed),online_loop_seconds=wall[name],
            final_state_sha256=model.state_digest()))
    return dict(seed=seed,metrics=metrics,diagnostics=diagnostics,reminders=reminders,damage=damage,
        failures=failures,costs=costs,capacity_references=capacity,development=development,
        stream={k:stream[k] for k in ['unique_identities','unique_inputs','distinct_input_outcome_pairs',
            'external','full_stream_sha256','prefix_check','prefix_costs']},
        routing=route.diagnostics(),seconds=time.perf_counter()-started)


SOURCE_REFERENCES = [
    ('fly_consolidation','Huang et al. 2024','https://www.nature.com/articles/s41586-024-07819-w'),
    ('silent_traces','Yang et al. 2026','https://www.nature.com/articles/s41593-026-02381-2'),
    ('MESU','Bonnet et al. 2025','https://arxiv.org/abs/2504.13569'),
    ('neuron_availability','Barry, Gerstner, Bellec','https://arxiv.org/abs/2306.01690'),
    ('SDM','Bricken et al. 2023','https://arxiv.org/abs/2303.11934'),
    ('GPM','Saha, Garg, Roy 2021','https://arxiv.org/abs/2103.09762'),
    ('SQHN','Alonso, Krichmar 2024','https://www.nature.com/articles/s41467-024-46976-4'),
    ('Continual_Backprop','Dohare et al. 2024','https://www.nature.com/articles/s41586-024-07711-7'),
    ('active_forgetting','Wang et al. 2023','https://arxiv.org/abs/2308.14991'),
    ('Progress_and_Compress','Schwarz et al. 2018','https://proceedings.mlr.press/v80/schwarz18a.html'),
    ('FlyModel','Shen, Dasgupta, Navlakha','https://arxiv.org/abs/2107.07617'),
    ('coupled_synapses','Benna and Fusi','https://arxiv.org/abs/1507.07580'),
    ('synaptic_intelligence','Zenke, Poole, Ganguli','https://arxiv.org/abs/1703.04200'),
    ('fast_weights','Ba et al.','https://arxiv.org/abs/1610.06258'),
    ('tiny_replay','Chaudhry et al.','https://arxiv.org/abs/1902.10486')]


def rows_csv(path, rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with open(path,'w',newline='',encoding='utf-8') as f:
        if keys:
            w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def atomic_json(path,obj):
    tmp=Path(str(path)+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')
    tmp.replace(path)


def summarize_results(results,cfg,specs):
    summary=[]
    for result in results:
        last={}
        for m in result['metrics']:
            last[m['arm'],m['phase']]=m
        for spec in specs:
            name=spec['name'];cost=next(c for c in result['costs'] if c['arm']==name)
            def get(phase,key):return last.get((name,phase),{}).get(key)
            phases=[m for m in result['metrics'] if m['arm']==name]
            terminal=phases[-1] if phases else {}
            summary.append(dict(seed=result['seed'],arm=name,family=spec['family'],status=cost['status'],
                A_acquired=get('acquire_A','A_current'),A_after_B=get('interfere_B','A_current'),
                B_acquired=get('interfere_B','B_current'),
                first_revision=get('revise_subset','phase_current'),
                v22_unchanged=get('revise_subset','initial_unchanged'),
                v22_all_current=get('revise_subset','all_current'),
                first_revision_after_C=get('interfere_C','initial_revised'),
                C_acquired=get('interfere_C','C_current'),
                second_revision=get('revise_again','phase_current'),
                short_final_current=get('revise_again','all_current'),
                terminal_phase=terminal.get('phase'),terminal_all_current=terminal.get('all_current'),
                terminal_phase_current=terminal.get('phase_current'),
                terminal_prior_current=terminal.get('prior_current'),
                total_labeled_presentations=cost['total_labeled_presentations'],
                peak_mutable_numeric_bytes=cost['peak_mutable_numeric_bytes']))
    comparisons=[]
    by={(r['seed'],r['arm']):r for r in summary}
    for r in summary:
        base=by.get((r['seed'],'ordinary'))
        if base and r['status']=='completed' and base['status']=='completed':
            d=dict(seed=r['seed'],arm=r['arm'],control='ordinary')
            for k in ['A_after_B','B_acquired','first_revision','first_revision_after_C',
                      'C_acquired','second_revision','terminal_all_current','terminal_prior_current']:
                if r.get(k) is not None and base.get(k) is not None:d[k+'_delta_pp']=100*(r[k]-base[k])
            comparisons.append(d)
    return summary,comparisons


def write_outputs(out,results,cfg,specs,verification,status):
    summary,comparisons=summarize_results(results,cfg,specs)
    rows_csv(out/'SUMMARY.csv',summary);rows_csv(out/'PAIRED_COMPARISONS.csv',comparisons)
    for key,filename in [('metrics','METRICS.csv'),('diagnostics','DIAGNOSTICS.csv'),
        ('reminders','FROZEN_REMINDERS.csv'),('damage','WRITE_EFFECTS.csv'),('costs','COSTS.csv'),
        ('failures','FAILURES.csv')]:
        rows_csv(out/filename,[r for x in results for r in x[key]])
    rows_csv(out/'CAPACITY_REFERENCES.csv',[dict(seed=x['seed'],**r) for x in results for r in x['capacity_references']])
    atomic_json(out/'VERIFICATION.json',verification)
    atomic_json(out/'STREAMS.json',[dict(seed=x['seed'],**x['stream']) for x in results])
    atomic_json(out/'DEVELOPMENT.json',[dict(seed=x['seed'],**x['development']) for x in results])
    atomic_json(out/'ARM_MANIFEST.json',specs)
    atomic_json(out/'ANATOMY_PROVENANCE.json',REDUCTION_PROVENANCE)
    failed=[r for x in results for r in x['failures']]
    def pct(v):return '—' if v is None else '%.2f'% (100*v)
    lines=['# ELM V23 consolidated mechanism screen','',
        'Status: **%s**. %d/%d seeds completed; %d arms; %d recorded arm failures.'%
        (status,len(results),len(cfg.seeds),len(specs),len(failed)),
        '','This is a screen of engineering mechanisms, not a reproduction or ranking of the source papers. '
        'The target remains persistent non-Transformer learning; the common ELM encoder is frozen online. '
        'Sparse address and recycling arms adapt additional downstream parameters, which are counted.',
        '', '## Endpoints (means of completed seeds; all seed rows retained)', '',
        '| Arm | n | A after B | B learned | Revision after C | Second revision | Terminal all | Mutable KiB |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for spec in specs:
        group=[r for r in summary if r['arm']==spec['name'] and r['status']=='completed']
        def mean(k):
            v=[r[k] for r in group if r.get(k) is not None]
            return float(np.mean(v)) if v else None
        vals=[pct(mean(k)) for k in ['A_after_B','B_acquired','first_revision_after_C','second_revision','terminal_all_current']]
        mem=mean('peak_mutable_numeric_bytes')
        lines.append('| %s | %d | %s | %s |'%(spec['name'],len(group),' | '.join(vals),
            '—' if mem is None else '%.1f'%(mem/1024)))
    lines += ['', 'Do not compare means with different completed-seed counts as if they were paired. '
        'Failures are not dropped from the run record. PAIRED_COMPARISONS.csv retains each valid seed contrast.',
        '', '## Protocol and interpretation', '',
        '- Original A/B/revision data and order preserve V22 RNG namespaces; STREAMS.json records prefix checks. '
        'C and a third-label revision follow, then the declared novel/revision stress waves. '
        'Cohort names and IDs belong only to the evaluator.',
        '- Every external event is predicted before its label enters observe(). Replay adds a charged presentation. '
        'Decay/consolidation advances once per external event, with no time ticks for probes or reminders.',
        '- The larger stress stream deliberately exceeds the 96-dimensional readout capacity in some regimes. '
        'Current-label offline ridge feasibility references are privileged diagnostics, never online contestants or accuracy upper bounds.',
        '- Extra addresses, slots, banks, covariance, importance and uncertainty are memory costs. '
        'This is not an equal-capacity comparison. Mutable bytes, declared fixed bytes, presentations and elapsed loop time are reported. '
        'Reported numeric bytes exclude Python object overhead and shared encoder/anatomy/evaluator arrays.',
        '- Historical ridge retains conflicting labels; strict projection can prevent revision; replay can preserve obsolete targets. '
        'Acquisition, retention, revision, and cost must be read together.',
        '- Frozen reminders supply unlabeled sensory information. Reminder identities are excluded from target evaluation. '
        'Matched and unrelated reminders leave storage, gate state, clock and RNG unchanged. '
        'Recovery of a superseded label is recorded separately and is not adaptation success. '
        'A and B are random cohorts without guaranteed distinguishable context, so a null reminder effect does not reject biological silent traces.',
        '- Sampled write effects compare net score changes, including decay and consolidation, against a hypothetical ordinary '
        'additive write on the same input feature and expressed residual. This is score displacement, not intrinsically harmful forgetting. '
        'Old-label MSE change is reported separately. Retrieval and allocation can be nonlinear; this is not an exact matched-parameter counterfactual for them.',
        '- SDM is represented by positive competitive address proxies, not merely a new top-k fraction. '
        'The quantized allocation arm is not full hierarchical MAP SQHN. Bayesian filtering is not exact MESU. '
        'Incremental projection has no task-boundary SVD. Availability alone is not the full context-selective Barry model. '
        'The active-forgetting proxy receives no task IDs, unlike published CAF experiments. '
        'Frozen-storage recovery is a computational diagnostic, not the biological recovery process.',
        '- No full trajectory dump or automatic hyperparameter selection is performed. '
        'A best-of-many result needs subsequent confirmation; the tested variants do not eliminate entire mechanism families.',
        '', '## Exposure totals', '']
    for result in results:
        stream=result['stream'];dev=result['development']
        lines.append('- Seed %d: %d identities, %d unique sensory training inputs, %d distinct input/outcome pairs, '
            '%d external presentations per arm; %d encoder-development training plus %d calibration presentations. '
            'Extra presentations across arms: %d.'%(result['seed'],stream['unique_identities'],stream['unique_inputs'],
                stream['distinct_input_outcome_pairs'],stream['external'],dev['development_training_presentations'],
                dev['development_calibration_presentations'],sum(c['extra_labeled_presentations'] for c in result['costs'])))
    lines += ['', 'FROZEN_REMINDERS.csv records additional reminder inputs; these are information exposures without labeled updates. '
        'Stored-state coupling operations are unlabeled arithmetic, not new unique examples. '
        'METRICS.csv includes both external and total labeled presentations at each checkpoint.',
        '', '## Sources and scope','']
    for mechanism,title,url in SOURCE_REFERENCES:
        lines.append('- %s: [%s](%s)'%(mechanism,title,url))
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    archive=out.with_name(out.name+'_results.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(out.iterdir()):
            # Seed resume checkpoints are intentionally not duplicated in ZIP.
            if p.is_file() and p.suffix in ('.csv','.json','.md'):
                z.write(p,p.name)
    return archive


def validation_checks(cfg, anatomy, specs):
    results={}
    results.update(weight_mechanism_tests())
    results.update(structural_mechanism_tests())
    results.update(replay_mechanism_tests())
    route=Route(anatomy,17,make_codebook(17))
    r=np.random.default_rng(2317)
    H=unit_rows(r.normal(size=(16,96)))
    # Every manifest entry must instantiate the advertised model, accept the
    # same API, preserve pure inference, and use the common external clock.
    for spec in specs:
        m=make_memory(spec,17,route,make_codebook(17),cfg)
        for i,h in enumerate(H):
            info=m.observe(h,i%4)
            if m.t!=i+1:raise AssertionError(spec['name']+' has incorrect clock')
            if info.get('extra_labeled_presentations',0) not in (0,1):
                raise AssertionError(spec['name']+' has unexpected replay budget')
        digest=m.state_digest()
        if np.asarray(m.output(H[:3])).shape!=(3,4):raise AssertionError('batch score shape')
        if np.asarray(m.output(H[0])).shape!=(4,):raise AssertionError('single score shape')
        if m.state_digest()!=digest:raise AssertionError('Inference changed '+spec['name'])
        if not np.isfinite(m.output(H)).all():raise AssertionError('Nonfinite '+spec['name'])
    results['all_manifest_arms_instantiate_and_obey_clock']=len(specs)
    # Stream identity uses standard V22 settings irrespective of smoke settings.
    prefix_cfg=argparse.Namespace(**vars(cfg))
    for k,v in dict(group_cues=24,variants=2,probe_variants=3,passes=3,
                    reverse_cues=12,noise=.06,stress_waves=0).items():setattr(prefix_cfg,k,v)
    prefix_checks=[build_stream(prefix_cfg,s)['prefix_check'] for s in [17,29,43]]
    if not all(x['status']=='matched' for x in prefix_checks):raise AssertionError('V22 prefix mismatch')
    results['v22_original_three_seed_input_label_order_hashes']=prefix_checks
    # Reference arithmetic agrees with an independent normalized delta update
    # and direct batch ridge solve on the accumulated numerical observations.
    u=LegacyMemory(96,17,route,make_codebook(17),cfg,'uniform_delta')
    ridge=LegacyMemory(96,17,route,make_codebook(17),cfg,'cumulative_ridge')
    W=np.zeros((96,4));seen=[];targets=[]
    for i,h in enumerate(H):
        y=target4(i%4)
        W+=cfg.learning_rate*np.outer(h,y-h@W)/(h@h)
        u.observe(h,i%4);ridge.observe(h,i%4);seen.append(h);targets.append(y)
    if not np.allclose(u.output(H),H@W,atol=1e-12):raise AssertionError('Ordinary reference differs')
    X=np.asarray(seen);Y=np.asarray(targets)
    batch=np.linalg.solve(cfg.ridge*np.eye(96)+X.T@X,X.T@Y)
    if not np.allclose(ridge.output(H),H@batch,atol=1e-10):raise AssertionError('Ridge differs from batch solve')
    results['normalized_delta_and_recursive_ridge_arithmetic']=True
    return results


def parse_config():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--install-deps',action='store_true',help='Install NumPy/PyTorch only if missing.')
    ap.add_argument('--out-dir',type=Path)
    ap.add_argument('--resume',type=Path,help='Resume from completed seed checkpoints in this directory; unfinished seed restarts.')
    ap.add_argument('--seeds',type=int,nargs='+',default=[17,29,43,59,71,89])
    ap.add_argument('--stress-waves',type=int,default=8,help='After V22+C+second revision, append this many new-group/revision pairs.')
    ap.add_argument('--dev-cues',type=int,default=64)
    ap.add_argument('--dev-passes',type=int,default=6)
    ap.add_argument('--group-cues',type=int,default=24)
    ap.add_argument('--variants',type=int,default=2)
    ap.add_argument('--probe-variants',type=int,default=3)
    ap.add_argument('--passes',type=int,default=3)
    ap.add_argument('--reverse-cues',type=int,default=12)
    ap.add_argument('--noise',type=float,default=.06)
    ap.add_argument('--active-fraction',type=float,default=.25)
    ap.add_argument('--learning-rate',type=float,default=.6,
                    help='Legacy reference rate; new mechanism rates are fixed in the arm definitions.')
    ap.add_argument('--write-energy-floor',type=float,default=.5)
    ap.add_argument('--ridge',type=float,default=.5)
    ap.add_argument('--threads',type=int,default=1)
    ap.add_argument('--diagnostic-interval',type=int,default=32,help='Sample write effects every N events;0 disables.')
    ap.add_argument('--arms',nargs='+',help='Optional exact arm names; default runs the complete manifest.')
    ap.add_argument('--list-arms',action='store_true')
    ap.add_argument('--self-test',action='store_true',help='Semantic/arithmetic checks only; no sensory encoder training.')
    ap.add_argument('--smoke',action='store_true',help='All arms, tiny complete workflow; validates execution only.')
    cfg=ap.parse_args();cfg.neurons=96
    if cfg.smoke:
        cfg.seeds=[17];cfg.dev_cues=16;cfg.dev_passes=1;cfg.group_cues=8
        cfg.passes=1;cfg.reverse_cues=4;cfg.stress_waves=1;cfg.diagnostic_interval=8
    if len(set(cfg.seeds))!=len(cfg.seeds):ap.error('Seeds must be distinct.')
    if cfg.group_cues<4 or cfg.group_cues%4:ap.error('--group-cues must be a positive multiple of4, at least4.')
    if cfg.reverse_cues<2 or cfg.reverse_cues%2 or cfg.reverse_cues>2*cfg.group_cues:
        ap.error('--reverse-cues must be even and between2 and twice group-cues.')
    if min(cfg.dev_cues,cfg.dev_passes,cfg.variants,cfg.probe_variants,cfg.passes,cfg.threads)<1:
        ap.error('Counts and threads must be positive.')
    if cfg.stress_waves<0 or cfg.diagnostic_interval<0:ap.error('Wave count and diagnostic interval cannot be negative.')
    if not 0<cfg.active_fraction<=1 or not 0<cfg.learning_rate<=1 or not 0<cfg.write_energy_floor<=1:
        ap.error('Fractions/rates must be in (0,1].')
    if cfg.ridge<=0 or cfg.noise<0:ap.error('Ridge must be positive and noise nonnegative.')
    if cfg.resume and cfg.out_dir:ap.error('Use either --resume or --out-dir.')
    return cfg


def main_v23():
    cfg=parse_config();specs=arm_manifest()
    if cfg.list_arms:
        for s in specs:print('%-30s %-26s %s'%(s['name'],s['family'],json.dumps(s['params'],sort_keys=True)))
        return 0
    if cfg.arms:
        wanted=set(cfg.arms);unknown=wanted-{s['name'] for s in specs}
        if unknown:raise SystemExit('Unknown arm(s): '+', '.join(sorted(unknown)))
        specs=[s for s in specs if s['name'] in wanted]
    # Set BLAS environment before importing NumPy; Torch has its own setting.
    for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
        os.environ[name]=str(cfg.threads)
    dependencies(cfg.install_deps)
    torch.set_num_threads(cfg.threads)
    try:torch.set_num_interop_threads(1)
    except RuntimeError:pass
    torch.use_deterministic_algorithms(True)
    anatomy=load_anatomy()
    print('ELM V23: %d arms; %d seeds; %d stress waves.'%(len(specs),len(cfg.seeds),cfg.stress_waves),flush=True)
    verification=validation_checks(cfg,anatomy,specs)
    print('Semantic, causality, V22 stream, and reference arithmetic checks passed.',flush=True)
    if cfg.self_test:
        print(json.dumps(verification,indent=2));return 0
    out=(cfg.resume or cfg.out_dir or Path('elm_fly_v23_'+time.strftime('%Y%m%d_%H%M%S')+'_'+str(os.getpid()))).expanduser().resolve()
    source_hash=sha_bytes(Path(__file__).read_bytes())
    args={k:(str(v) if isinstance(v,Path) else v) for k,v in vars(cfg).items()}
    identity={k:v for k,v in args.items() if k not in ('out_dir','resume','install_deps','list_arms','self_test')}
    identity.update(source_sha256=source_hash,arms=specs)
    fingerprint=sha_bytes(json.dumps(identity,sort_keys=True).encode())
    results=[]
    if cfg.resume:
        prior=json.loads((out/'CONFIG.json').read_text())
        if prior.get('experiment_fingerprint')!=fingerprint:
            raise SystemExit('Resume configuration/source differs. Reuse the original arguments and script, or start a new output directory.')
        for seed in cfg.seeds:
            path=out/'seed_checkpoints'/('seed_%d.json'%seed)
            if path.exists():results.append(json.loads(path.read_text()))
    else:
        out.mkdir(parents=True,exist_ok=False)
    (out/'seed_checkpoints').mkdir(exist_ok=True)
    config=dict(args,version=V23_VERSION,source_sha256=source_hash,source_v22_sha256=SOURCE_V22_SHA256,
        experiment_fingerprint=fingerprint,python=sys.version,numpy=np.__version__,torch=torch.__version__,
        platform=platform.platform(),arm_count=len(specs),
        independent_review='Family tests and integration checks; no completed independent full-code audit is claimed.')
    atomic_json(out/'CONFIG.json',config)
    print('Output: '+str(out),flush=True)
    print('Expected external presentations per arm per seed: %d'%
          ((3*cfg.group_cues+2*cfg.reverse_cues+cfg.stress_waves*(cfg.group_cues+cfg.reverse_cues))*cfg.variants*cfg.passes),flush=True)
    done={r['seed'] for r in results}
    status='completed'
    try:
        for seed in cfg.seeds:
            if seed in done:
                print('Reusing completed seed %d'%seed,flush=True);continue
            print('Seed %d'%seed,flush=True)
            result=run_experiment_seed(cfg,seed,anatomy,specs)
            atomic_json(out/'seed_checkpoints'/('seed_%d.json'%seed),result)
            results.append(result)
            archive=write_outputs(out,results,cfg,specs,verification,'running' if len(results)<len(cfg.seeds) else 'completed')
    except KeyboardInterrupt:
        status='interrupted_after_completed_seed_checkpoints'
        print('\nInterrupted. Completed seeds remain resumable; the unfinished seed will restart.',flush=True)
    except Exception as exc:
        status='runner_failed_'+type(exc).__name__
        atomic_json(out/'RUNNER_ERROR.json',{'exception':type(exc).__name__,'message':str(exc)})
        print('Runner error: %s: %s'%(type(exc).__name__,exc),flush=True)
    if any(r['failures'] for r in results) and status=='completed':status='completed_with_arm_failures'
    archive=write_outputs(out,results,cfg,specs,verification,status)
    print('Results ZIP: '+str(archive),flush=True)
    print('Status: '+status,flush=True)
    return 0 if status=='completed' else 2


VERSION = '24.0.0'


# === V24 encoder ===
import copy as _v24_copy


class V24Encoder:
    """Temporal scalar-cell encoder with explicit, bounded local plasticity.

    Local development: each dendritic branch uses its own signed branch
    response b and contacted sensory value x. A receiving-cell activity gate
    g=max(0,a-mean_population_activity) is supplied by the declared inhibitory
    population signal. dw=rate*g*(b*x-b*b*w); each weight clips to [-2,2].
    Intrinsic threshold follows rate*.25*(a-.15), clipped to [-5,3].
    No readout error, readout transpose or temporal computational graph enters
    either rule. Prescribed type parameters remain fixed and cells retain six
    separate private states. The local recurrent option uses the same rule on
    its own delayed presynaptic activity. All changes are per sensory time bin.
    """

    def __init__(self, cfg, seed, regime, model, mean=None, sd=None):
        self.regime, self.seed = regime, int(seed)
        self.N = int(cfg.neurons)
        self.variant = dict(online_rate=0., stable_fraction=0., isolate_fraction=0.,
            timescales='mixed', gates='ordinary', stubborn_fraction=0.,
            active_fraction=float(cfg.active_fraction), activity_code='signed',
            competition='exact', recurrence='pair', recurrent_learning=False,
            inhibitory_fixed=False, inhibitory_learning=False, turnover='none', reset='experience',
            precision='float64')
        self.model = model if regime in ('E0', 'E1') else None
        self.arrays = {}
        if self.model is not None:
            self.model.eval()
        else:
            for name, a in model.state_dict().items():
                v = a.detach().cpu().numpy()
                self.arrays[name] = v.astype(np.int64 if v.dtype.kind in 'iu' else np.float64).copy()
        self.mean = np.zeros(self.N) if mean is None else np.asarray(mean).copy()
        self.sd = np.ones(self.N) if sd is None else np.asarray(sd).copy()
        self.private_state = np.zeros((self.N, 6))
        self.private_history = np.zeros((3, self.N))
        self.local_steps = 0
        self.sensory_presentations = 0
        self.rewired_contacts = 0
        self.inhibitory_sources = (np.arange(self.N) + 1) % self.N
        self.inhibitory_weights = np.zeros(self.N)
        self.activity_rate = np.full(self.N, .15)
        self.synapse_q8 = None
        self.quantization_scale = 2. / 127.
        self.candidate_sources = None
        self.candidate_scores = None
        self.contact_scores = None
        self.development = {}

    def clone(self):
        return _v24_copy.deepcopy(self)

    copy = clone

    def configure(self, variant_dict):
        if not variant_dict:
            return self
        unknown = set(variant_dict) - set(self.variant)
        if unknown:
            raise ValueError('Unknown encoder variant keys: '+','.join(sorted(unknown)))
        if self.model is not None and variant_dict:
            raise ValueError('Circuit/online variants require E2 or E3, not a BP reference.')
        self.variant.update(variant_dict)
        v = self.variant
        for key, allowed in dict(timescales=('mixed','single'), gates=('ordinary','coupled','independent'),
                activity_code=('signed','nonnegative'), competition=('exact','inhibitory'),
                recurrence=('pair','pooled','coincidence','off'), turnover=('none','retained','local'),
                reset=('experience','continuous'), precision=('float64','float32','quantized8')).items():
            if v[key] not in allowed:
                raise ValueError('Unsupported encoder '+key+': '+str(v[key]))
        if not 0 < v['active_fraction'] <= 1 or not 0 <= v['stable_fraction'] < 1:
            raise ValueError('Encoder activity/stability fractions outside valid range.')
        if v['online_rate'] < 0:
            raise ValueError('Negative local learning rate.')
        if (v['recurrent_learning'] or v['inhibitory_learning'] or v['turnover']=='local') and 'online_rate' not in variant_dict:
            v['online_rate'] = .0002
        isolate = v['isolate_fraction'] or v['stable_fraction']
        if isolate:
            # Both partitions have closed recurrent inputs. Pair with the same
            # isolate_fraction and stable_fraction=0 to control graph changes.
            boundary = max(9, min(self.N-9, round(self.N*isolate)))
            rg = rng_for(self.seed, 'v24_isolated_graph')
            for i in range(self.N):
                pool = np.arange(boundary) if i < boundary else np.arange(boundary,self.N)
                pool = pool[pool != i]
                self.arrays['recurrent'][i] = rg.choice(pool,8,replace=False)
            self.inhibitory_sources[:boundary] = (np.arange(boundary)+1) % boundary
            self.inhibitory_sources[boundary:] = boundary+(np.arange(self.N-boundary)+1) % (self.N-boundary)
            v['isolate_fraction'] = boundary / self.N
        if v['competition']=='inhibitory' or v['inhibitory_fixed'] or v['inhibitory_learning']:
            self.inhibitory_weights[:] = .05
        if v['turnover'] in ('retained','local') and self.candidate_sources is None:
            rg = rng_for(self.seed,'v24_candidate_contacts')
            self.candidate_sources = rg.integers(0,32,size=(self.N,8,4))
            self.candidate_scores = np.zeros((self.N,8,4))
            self.contact_scores = np.zeros((self.N,8,3))
        dtype = np.float64 if v['precision']=='float64' else np.float32
        for name, a in list(self.arrays.items()):
            if a.dtype.kind == 'f': self.arrays[name] = a.astype(dtype)
        for name in ('mean','sd','private_state','private_history','inhibitory_weights','activity_rate'):
            setattr(self,name,getattr(self,name).astype(dtype))
        if v['precision']=='quantized8':
            if self.synapse_q8 is None:
                self.synapse_q8 = np.round(np.clip(self.arrays.pop('syn'),-2,2)/self.quantization_scale).astype(np.int8)
        elif self.synapse_q8 is not None:
            self.arrays['syn'] = (self.synapse_q8.astype(dtype)*self.quantization_scale)
            self.synapse_q8 = None
        self.private_state.fill(0); self.private_history.fill(0)
        return self

    def _syn(self):
        if self.synapse_q8 is None:
            return self.arrays['syn']
        return self.synapse_q8.astype(np.float32)*self.quantization_scale

    def _write_syn(self, syn):
        np.clip(syn,-2.,2.,out=syn)
        if self.synapse_q8 is None:
            self.arrays['syn'][...] = syn
        else:
            self.synapse_q8[...] = np.rint(syn/self.quantization_scale).astype(np.int8)

    def _forward_numpy(self, x, adapt=False, learning_rate=None):
        a = self.arrays; v = self.variant
        dtype = a['wm'].dtype
        x = np.asarray(x,dtype=dtype)
        B,T,D = x.shape
        if (T,D)!=(12,32):
            raise ValueError('Encoder requires twelve bins of 32 observed values.')
        if adapt and B != 1:
            raise ValueError('Causal encoder adaptation consumes one experience at a time.')
        if v['reset']=='continuous':
            m = np.repeat(self.private_state[None],B,axis=0)
            hist = np.repeat(self.private_history[None],B,axis=0)
        else:
            m = np.zeros((B,self.N,6),dtype=dtype)
            hist = np.zeros((B,3,self.N),dtype=dtype)
        tau = np.exp(math.log(1.5)+math.log(32./1.5)/(1.+np.exp(-a['tau_raw'])))
        alpha = -np.expm1(-1./tau)
        if v['timescales']=='single':
            # Match mean state-update rate, not arithmetic mean time constant.
            alpha = np.full_like(alpha,float(alpha.mean()))
        gain = np.logaddexp(0.,a['gain'])
        w1,w2 = a['w1'][a['type_index']], a['w2'][a['type_index']]
        b1,b2 = a['b1'][a['type_index']], a['b2'][a['type_index']]
        syn = self._syn().copy() if adapt else self._syn()
        histidx = ((a['delays']-1)*self.N+a['recurrent']).reshape(-1)
        plast = np.ones(self.N,dtype=dtype)
        plast[:round(self.N*v['stable_fraction'])] = 0
        rate = float(v['online_rate'] if learning_rate is None else learning_rate)
        do_learn = adapt and rate > 0
        for t in range(T):
            sensed = x[:,t][:,a['sensory']]
            rec = hist.reshape(B,-1)[:,histidx].reshape(B,self.N,8)
            if v['recurrence']=='off': rec = np.zeros_like(rec)
            elif v['recurrence']=='pooled': rec = np.broadcast_to(rec.mean(-1,keepdims=True),rec.shape)
            elif v['recurrence']=='coincidence': rec = rec*np.tanh(np.abs(sensed).mean(-1))
            branch = np.tanh(gain*((sensed*syn[None,:,:,:3]).sum(-1)+rec*syn[None,:,:,3]+a['bias']))
            z = np.concatenate((branch,m),axis=-1)
            hidden = np.tanh(np.einsum('bnq,nqh->bnh',z,w1,optimize=False)+b1)
            candidate = np.tanh(np.einsum('bnh,nhm->bnm',hidden,w2,optimize=False)+b2)
            write = alpha[None]
            forget = alpha[None]
            if v['stubborn_fraction']:
                n = round(self.N*v['stubborn_fraction'])
                write = np.broadcast_to(write,m.shape).copy()
                write[:,:n] /= 1.+4.*np.abs(candidate[:,:n]-m[:,:n])
                forget = write
            if v['gates']=='coupled':
                gate = .5+.5*np.tanh(np.abs(branch).mean(-1,keepdims=True))
                write = write*gate; forget = write
            elif v['gates']=='independent':
                incoming = .5+.5*np.tanh(np.abs(branch).mean(-1,keepdims=True))
                retaining = .5+.5*np.tanh(np.abs(m))
                write = write*incoming; forget = forget*retaining
            m = np.clip((1.-forget)*m+write*candidate,-1.,1.)
            voltage = (m*a['wm']).sum(-1)+(branch*a['wd']).sum(-1)+a['abias']
            if v['competition']=='inhibitory' or v['inhibitory_fixed'] or v['inhibitory_learning']:
                voltage = voltage - hist[:,0,self.inhibitory_sources]*self.inhibitory_weights
            threshold = np.logaddexp(0.,a['threshold'])
            activity = 1./(1.+np.exp(np.clip(-4.*(voltage-threshold),-60,60)))
            if do_learn:
                # Inhibition broadcasts only present population activity.
                gate = np.maximum(activity[0]-activity[0].mean(),0.)
                gate = np.minimum(8.*gate,1.)*plast
                post = branch[0]
                pre = sensed[0]
                syn[:,:,:3] += rate*gate[:,None,None]*(post[:,:,None]*pre-post[:,:,None]**2*syn[:,:,:3])
                if v['recurrent_learning']:
                    syn[:,:,3] += rate*gate[:,None]*(post*rec[0]-post**2*syn[:,:,3])
                np.clip(syn,-2,2,out=syn)
                a['threshold'] += rate*.25*plast*(activity[0]-.15)
                np.clip(a['threshold'],-5,3,out=a['threshold'])
                self.activity_rate += .01*(activity[0]-self.activity_rate)
                if v['inhibitory_learning']:
                    # Increased coincident excess activity strengthens its
                    # receiving-cell inhibitory contact; no covariance solver.
                    self.inhibitory_weights += rate*plast*hist[0,0,self.inhibitory_sources]*(activity[0]-.15)
                    np.clip(self.inhibitory_weights,0,1,out=self.inhibitory_weights)
                if self.candidate_sources is not None:
                    cpre = x[0,t,self.candidate_sources]
                    self.candidate_scores *= .995
                    self.candidate_scores += .005*np.abs(post[:,:,None]*cpre)
                    self.contact_scores *= .995
                    self.contact_scores += .005*np.abs(post[:,:,None]*pre)
                self.local_steps += 1
            hist = np.concatenate((activity[:,None],hist[:,:2]),axis=1)
        if adapt:
            if do_learn: self._write_syn(syn)
            self.sensory_presentations += 1
            if v['reset']=='continuous':
                self.private_state[...] = m[0]
                self.private_history[...] = hist[0]
            if do_learn and v['turnover']=='local' and self.sensory_presentations % 48 == 0:
                # One local branch contact per cell competes with its bounded
                # candidate contacts. Retained control pays identical traces.
                for i in range(self.N):
                    if not plast[i]: continue
                    r = (self.sensory_presentations//48+i) % 8
                    old = int(np.argmin(self.contact_scores[i,r]))
                    new = int(np.argmax(self.candidate_scores[i,r]))
                    source = int(self.candidate_sources[i,r,new])
                    if (source not in a['sensory'][i,r] and
                            self.candidate_scores[i,r,new] > self.contact_scores[i,r,old]):
                        a['sensory'][i,r,old] = source
                        syn[i,r,old] = .1
                        self.contact_scores[i,r,old] = self.candidate_scores[i,r,new]
                        self.rewired_contacts += 1
                self._write_syn(syn)
        return activity

    def activities(self, x, adapt=False, learning_rate=None):
        x = np.asarray(x)
        if x.ndim == 2: x = x[None]
        if self.model is not None:
            if adapt:
                raise ValueError('BP-developed reference encoder stays frozen online.')
            return encode(self.model,x.astype(np.float32))
        if adapt:
            return np.concatenate([self._forward_numpy(row[None],True,learning_rate) for row in x])
        return self._forward_numpy(x,False)

    def features(self, x, feature='sparse', adapt=False):
        activity = self.activities(x,adapt=adapt)
        if feature not in ('sparse','dense'):
            raise ValueError('Unknown feature representation '+str(feature))
        v = self.variant
        z = np.clip((activity-self.mean)/self.sd,-5.,5.).astype(np.float64)
        if v['activity_code']=='nonnegative': z = np.maximum(z,0.)
        if v['isolate_fraction']:
            boundary = round(self.N*v['isolate_fraction'])
            # A shared top-k or global norm would let plastic activity alter
            # otherwise stable readout coordinates. The matched all-plastic
            # partition control uses exactly the same two local populations.
            output = []
            for left,right in ((0,boundary),(boundary,self.N)):
                part = self._expression(z[:,left:right].copy(),feature)
                part *= math.sqrt((right-left)/self.N)
                output.append(part)
            z = np.concatenate(output,axis=1)
        else:
            z = self._expression(z,feature)
        if v['precision'] in ('float32','quantized8'):
            z = z.astype(np.float32)
        return z

    def _expression(self, z, feature):
        v = self.variant
        if feature=='sparse':
            if v['competition']=='exact':
                k = max(1,round(z.shape[1]*v['active_fraction']))
                chosen = np.argpartition(np.abs(z),-k,axis=1)[:,-k:]
                h = np.zeros_like(z)
                np.put_along_axis(h,chosen,np.take_along_axis(z,chosen,axis=1),axis=1)
                z = h
            else:
                # A single nonnegative inhibitory population state receives
                # total excitation and feeds back subtraction. Twelve causal
                # relaxation steps; realized density is measured, not forced k.
                drive = np.maximum(z,0.)
                inhibition = np.maximum(drive.mean(1,keepdims=True),0.)
                for _ in range(12):
                    rate = np.maximum(drive-inhibition,0.)
                    inhibition = np.maximum(0.,inhibition+.12*(rate.mean(1,keepdims=True)-.20*v['active_fraction']))
                z = np.maximum(drive-inhibition,0.)
        z /= np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-12)
        return z

    def state_digest(self):
        h = hashlib.sha256()
        h.update(json.dumps(self.variant,sort_keys=True).encode())
        if self.model is not None:
            for name,a in self.model.state_dict().items():
                h.update(name.encode()); h.update(a.detach().cpu().numpy().tobytes())
        for name,a in sorted(self.arrays.items()):
            h.update(name.encode()); h.update(a.tobytes())
        for name in ('mean','sd','private_state','private_history','inhibitory_sources',
                     'inhibitory_weights','activity_rate','synapse_q8','candidate_sources',
                     'candidate_scores','contact_scores'):
            a = getattr(self,name)
            if a is not None:
                h.update(name.encode());h.update(a.tobytes())
        h.update(str((self.local_steps,self.sensory_presentations,self.rewired_contacts)).encode())
        return h.hexdigest()

    def costs(self):
        extra = {name:getattr(self,name) for name in ('mean','sd','private_state','private_history',
            'inhibitory_sources','inhibitory_weights','activity_rate','synapse_q8',
            'candidate_sources','candidate_scores','contact_scores') if getattr(self,name) is not None}
        arrays = dict(self.arrays)
        if self.model is not None:
            arrays.update({n:a.detach().cpu().numpy() for n,a in self.model.state_dict().items()})
        arrays.update(extra)
        actual = int(sum(a.nbytes for a in arrays.values()))
        live = self.variant['online_rate'] > 0
        mutable_names = {'private_state','private_history'} if self.variant['reset']=='continuous' else set()
        if live:
            mutable_names |= {'syn','synapse_q8','threshold','activity_rate'}
            if self.variant['inhibitory_learning']: mutable_names.add('inhibitory_weights')
            if self.candidate_sources is not None: mutable_names |= {'candidate_scores','contact_scores'}
            if self.variant['turnover']=='local': mutable_names.add('sensory')
        mutable = int(sum(a.nbytes for n,a in arrays.items() if n in mutable_names))
        return dict(encoder_array_bytes=actual,encoder_fixed_array_bytes=actual-mutable,
            encoder_mutable_array_bytes=mutable,encoder_private_state_values=int(self.N*9),
            encoder_synapse_storage_dtype=str(self.synapse_q8.dtype if self.synapse_q8 is not None else arrays['syn'].dtype),
            encoder_precision=self.variant['precision'],encoder_local_time_bins=self.local_steps,
            encoder_sensory_presentations=self.sensory_presentations,
            encoder_rewired_contacts=self.rewired_contacts,
            encoder_array_accounting_excludes='Python objects, temporary forward work arrays and scalar metadata',
            encoder_fixed_microdynamics='Four prescribed type templates; six independent states per cell',
            encoder_quantization_step=self.quantization_scale if self.synapse_q8 is not None else 0.)


def v24_develop_encoder(cfg,seed,regime):
    if regime not in ('E0','E1','E2','E3'):
        raise ValueError('Unknown encoder regime '+str(regime))
    if regime=='E0':
        model,mean,sd,info = develop_encoder(cfg,seed)
        enc = V24Encoder(cfg,seed,regime,model,mean,sd)
        info = dict(info)
        info.update(development_target='clean_latent_pulses_projected_64_to_16',
            hidden_backpropagation=True,teacher_values_per_example=16,
            local_rule='historical backprop reference', development_target_privilege=True)
    else:
        model = make_elm(cfg,seed)
        rg = rng_for(seed,'development_data')
        bases = cue_bases(rg,cfg.dev_cues)
        x = render_cues(bases,1,rg,cfg.noise)[:,0]
        losses=[]
        if regime=='E1':
            proj = rng_for(seed,'development_target').normal(size=(64,16))/8.
            # Read only the noisy values presented at bins 0 and 5. These are
            # copied over three bins; copies are not independent observations.
            observed = np.concatenate((x[:,0],x[:,5]),axis=-1)
            y = (observed@proj).astype(np.float32)
            torch.manual_seed(seed_for(seed,'development_head'))
            head = nn.Linear(cfg.neurons,16)
            optimizer = torch.optim.AdamW(list(model.parameters())+list(head.parameters()),lr=.003,weight_decay=1e-4)
            tx,ty = torch.from_numpy(x),torch.from_numpy(y)
            order_rng = rng_for(seed,'development_order')
            for epoch in range(cfg.dev_passes):
                model.train(); loss_sum=0.
                for idx in np.array_split(order_rng.permutation(len(x)),math.ceil(len(x)/16)):
                    optimizer.zero_grad(set_to_none=True)
                    activity=model(tx[idx]);pred=head(activity)
                    loss=F.mse_loss(pred,ty[idx])+.001*(activity.mean()-.15).square()
                    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.)
                    optimizer.step();loss_sum += float(loss.detach())*len(idx)
                losses.append(loss_sum/len(x))
            for p in model.parameters(): p.requires_grad_(False)
            calibration = encode(model,x)
            enc = V24Encoder(cfg,seed,regime,model,calibration.mean(0),np.maximum(calibration.std(0),.01))
            head_params = sum(p.numel() for p in head.parameters())
        else:
            enc = V24Encoder(cfg,seed,regime,model)
            head_params=0
            if regime=='E3':
                order_rng = rng_for(seed,'development_order')
                dev_rate = float(getattr(cfg,'local_dev_rate',.003))
                for epoch in range(cfg.dev_passes):
                    for idx in order_rng.permutation(len(x)):
                        enc.activities(x[idx],adapt=True,learning_rate=dev_rate)
            calibration = enc.activities(x,adapt=False)
            enc.mean=calibration.mean(0)
            enc.sd=np.maximum(calibration.std(0),.01)
        info=dict(loss_by_pass=losses,development_unique_inputs=len(x),
            development_training_presentations=len(x)*cfg.dev_passes if regime!='E2' else 0,
            development_calibration_presentations=len(x),outcome_labels_used=0,
            encoder_parameters=sum(p.numel() for p in model.parameters()),
            temporary_development_head_parameters=head_params,
            final_activity_std_median=float(np.median(calibration.std(0))),
            development_target='observed_noisy_pulses_projected_64_to_16' if regime=='E1' else 'sensory_activity_only',
            hidden_backpropagation=regime=='E1', teacher_values_per_example=16 if regime=='E1' else 0,
            development_target_privilege=False,
            local_rule='receiving branch Oja; local cell threshold; population inhibitory gate' if regime=='E3' else 'none',
            local_development_rate=float(getattr(cfg,'local_dev_rate',.003)) if regime=='E3' else 0.)
    info.update(encoder_regime=regime,local_updates_per_presentation=12 if regime=='E3' else 0,
        development_independent_noisy_views_per_identity=1,
        developmental_teacher_source='synthetic clean latent values' if regime=='E0' else
            ('observed sensory values only' if regime=='E1' else 'no vector target'),
        temporary_teacher_projection_values=1024 if regime in ('E0','E1') else 0,
        temporary_teacher_projection_bytes=8192 if regime in ('E0','E1') else 0,
        temporary_teacher_target_values=16*cfg.dev_cues if regime in ('E0','E1') else 0,
        online_synaptic_backpropagation=False,
        calibration_policy='frozen per-neuron mean and sd from separately counted development exposure',
        code_unit_normalization='explicit population normalization; not claimed synapse-local arithmetic',
        default_reset_policy='reset six private states and three delay histories each independent experience')
    enc.development=dict(info)
    info['encoder_sha256']=enc.state_digest()
    return enc,info


def v24_encoder_checks():
    from types import SimpleNamespace
    cfg=SimpleNamespace(neurons=96,active_fraction=.25,dev_cues=4,dev_passes=1,noise=.12)
    seed=17
    model=make_elm(cfg,seed)
    rg=rng_for(seed,'v24_encoder_check')
    x=rg.normal(0,.2,(3,12,32)).astype(np.float32)
    local=V24Encoder(cfg,seed,'E2',model)
    ref=encode(model,x)
    numerical=float(np.max(np.abs(ref-local.activities(x))))
    assert numerical < 2e-6, ('NumPy temporal conversion differs',numerical)
    before=local.state_digest();local.features(x);assert before==local.state_digest()
    frozen=local.clone();frozen.configure(dict(reset='continuous'))
    before=frozen.state_digest();frozen.features(x);assert before==frozen.state_digest()
    frozen.features(x[0],adapt=True);assert before!=frozen.state_digest()
    dev,meta=v24_develop_encoder(cfg,seed,'E3')
    initial=V24Encoder(cfg,seed,'E2',make_elm(cfg,seed))
    assert not np.array_equal(initial.arrays['syn'],dev.arrays['syn'])
    assert np.array_equal(initial.arrays['w1'],dev.arrays['w1'])
    assert np.array_equal(initial.arrays['syn'][:,:,3],dev.arrays['syn'][:,:,3])
    variant_checks=[dict(timescales='single'),dict(gates='coupled'),dict(gates='independent'),
        dict(stubborn_fraction=.5),dict(active_fraction=.125),dict(active_fraction=.5),
        dict(activity_code='nonnegative'),dict(activity_code='nonnegative',competition='inhibitory'),
        dict(recurrence='pooled'),dict(recurrence='coincidence'),dict(recurrent_learning=True),
        dict(inhibitory_learning=True),dict(turnover='retained',online_rate=.0002),
        dict(turnover='local',online_rate=.0002),dict(precision='float32'),dict(precision='quantized8')]
    for variant in variant_checks:
        e=dev.clone().configure(variant)
        before=e.state_digest();h=e.features(x);assert np.isfinite(h).all();assert before==e.state_digest()
        h=e.features(x[0],adapt=True);assert np.isfinite(h).all()
    fixed_inhib=dev.clone().configure(dict(online_rate=.0002,inhibitory_fixed=True))
    learned_inhib=dev.clone().configure(dict(online_rate=.0002,inhibitory_fixed=True,inhibitory_learning=True))
    assert np.array_equal(fixed_inhib.features(x),learned_inhib.features(x))
    assert fixed_inhib.costs()['encoder_array_bytes']==learned_inhib.costs()['encoder_array_bytes']
    fixed_contacts=fixed_inhib.inhibitory_weights.copy()
    for row in x:
        fixed_inhib.features(row,adapt=True); learned_inhib.features(row,adapt=True)
    assert np.array_equal(fixed_inhib.inhibitory_weights,fixed_contacts)
    assert not np.array_equal(learned_inhib.inhibitory_weights,fixed_contacts)
    split=dev.clone().configure(dict(stable_fraction=.5,online_rate=.001))
    first=split.activities(x)
    first_features=split.features(x)
    for row in x: split.features(row,adapt=True)
    second=split.activities(x)
    assert np.max(np.abs(first[:,:48]-second[:,:48])) < 1e-12
    assert np.max(np.abs(first_features[:,:48]-split.features(x)[:,:48])) < 1e-12
    return dict(numpy_reference_max_error=numerical,pure_probe_verified=True,
        local_development_changes_afferents=True,prescribed_microdynamics_unchanged=True,
        frozen_recurrent_weights_unchanged=True,variant_checks=len(variant_checks),
        matched_fixed_inhibitory_contact_control_verified=True,
        isolated_stable_population_verified=True,local_development_presentations=meta['development_training_presentations'])


# === V24 memory ===

# V24 memory mechanisms. These classes never receive cue identities or phases.
# Exact sorting/normalization and historical buffers are declared algorithmic
# primitives, not automatically claimed to be a complete biological circuit.

def v24_memory_array_bytes(model):
    """Persistent NumPy arrays only, deduplicated; scalar/RNG costs are separate."""
    arrays = dict(model.mutable_arrays())
    arrays.update({'fixed.' + k: v for k, v in
                   (model.fixed_arrays() if hasattr(model, 'fixed_arrays') else {}).items()})
    seen, total = set(), 0
    for array in arrays.values():
        root = array
        while isinstance(getattr(root, 'base', None), np.ndarray):
            root = root.base
        if id(root) not in seen:
            seen.add(id(root)); total += int(root.nbytes)
    return total


def v24_competition(scores):
    """Fixed pairwise comparator tree; ties choose the lower-index candidate.

    Each stage communicates only with its paired neighbor. This is an explicit
    engineering competition circuit, not an anatomical identification. Return
    the winning endpoint; it uses no learned downstream derivative.
    """
    score = np.asarray(scores, dtype=np.float64).reshape(-1).copy()
    if not len(score):
        raise ValueError('Competition needs at least one candidate')
    indices = np.arange(len(score), dtype=np.int64)
    while len(score) > 1:
        n = len(score) // 2
        choose_right = score[1:2*n:2] > score[:2*n:2]
        s = np.where(choose_right, score[1:2*n:2], score[:2*n:2])
        ix = np.where(choose_right, indices[1:2*n:2], indices[:2*n:2])
        if len(score) % 2:
            s = np.concatenate((s, score[-1:])); ix = np.concatenate((ix, indices[-1:]))
        score, indices = s, ix
    return int(indices[0])


class V24BankMemory(MemoryBase):
    """Common-initialization content banks and independent pure read views.

    All banks are present from initialization. The write, trace release, and
    sensory prototype adaptation always use the selected bank's local state.
    Neither soft retrieval nor the temporary recent-access prior influences
    these supervised updates. Fixed and adaptive routers start identically.

    A coefficient budget stores genuinely compact readouts: the V property
    reconstructs a temporary dense evaluator view. Only stored coefficients
    appear in mutable_arrays; no zero-padded unused weights are charged as a
    capacity control. The exposed feature-index wiring is a fixed prior.
    """
    def __init__(self, d, seed, banks=8, routing='fixed', eta=.6,
                 spillover=0., coefficient_budget=None, protection='none',
                 release=False, scramble=False, strength=3., recovery=.997,
                 address_eta=.05, temperature=8., read_mode='hard',
                 recent_bias=0., **extra):
        super().__init__(d, seed)
        self.banks = int(banks)
        if self.banks < 1:
            raise ValueError('Bank count must be positive')
        self.routing = {'learned': 'local', 'adaptive': 'local'}.get(routing, routing)
        if self.routing not in ('fixed', 'local'):
            raise ValueError('Common-initialization bank routing is fixed or local')
        self.eta = float(eta); self.spillover = float(spillover)
        if not 0 <= self.spillover <= 1:
            raise ValueError('Spillover must be in [0,1]')
        self.protection = str(protection); self.release = bool(release)
        self.scramble = bool(scramble); self.strength = float(strength)
        self.recovery = float(recovery)
        if self.protection not in ('none', 'neuron', 'pair'):
            raise ValueError('Protection must be none, neuron, or pair')
        self.address_eta = float(address_eta); self.temperature = float(temperature)
        self.read_mode = str(read_mode); self.recent_bias = float(recent_bias)
        self.coefficient_budget = (None if coefficient_budget is None
                                   else int(coefficient_budget))
        self.width = (self.d if self.coefficient_budget is None else
                      min(self.d, self.coefficient_budget // (4*self.banks)))
        if self.width < 1:
            raise ValueError('Coefficient budget cannot provide one input per bank')
        # Router draws occur before optional protection/wiring draws, so bank
        # interventions preserve the same sensory competition initial state.
        self.centers = unit_rows(self.rng.normal(size=(self.banks, self.d)))
        self._V = np.zeros((self.banks, self.width, 4), dtype=np.float64)
        self.feature_index = None
        if self.width != self.d:
            order = self.rng.permutation(self.d)
            self.feature_index = np.stack([
                np.roll(order, -b*self.width)[:self.width] for b in range(self.banks)
            ]).astype(np.int64)
        self.counts = np.zeros(self.banks, dtype=np.int64)
        self.last_seen = np.zeros(self.banks, dtype=np.int64)
        self.active = -1; self.used = self.banks; self.route_changes = 0
        self.releases = 0; self.gain_sum = 0.; self.correction_sum = 0.
        self.empty_write_events = 0
        self.importance = None; self.surprise_ema = None; self.permutation = None
        if self.protection != 'none':
            channels = 1 if self.protection == 'neuron' else 4
            self.importance = np.zeros((self.banks, self.width, channels))
            self.surprise_ema = np.ones(self.banks)
            # Both intact and shuffled protection allocate the fixed routing
            # table. Intact uses identity wiring; shuffled preserves each
            # output's trace multiset while breaking its connection pairing.
            self.permutation = np.tile(np.arange(self.width)[None, :, None],
                                       (self.banks, 1, channels))
            if self.scramble:
                for b in range(self.banks):
                    for c in range(channels):
                        self.permutation[b, :, c] = self.rng.permutation(self.width)

    @property
    def V(self):
        if self.feature_index is None:
            return self._V
        dense = np.zeros((self.banks, self.d, 4), dtype=self._V.dtype)
        for b in range(self.banks):
            dense[b, self.feature_index[b]] = self._V[b]
        return dense

    @V.setter
    def V(self, value):
        value = np.asarray(value)
        if self.feature_index is None:
            self._V[...] = value
        else:
            for b in range(self.banks):
                self._V[b] = value[b, self.feature_index[b]]

    def _bank_input(self, H, bank):
        return H if self.feature_index is None else H[..., self.feature_index[bank]]

    def route_probabilities(self, H, read_mode='hard'):
        H, scalar = _structural_rows(H)
        score = unit_rows(H) @ self.centers.T
        if read_mode in ('hard', 'winner'):
            p = np.zeros_like(score)
            p[np.arange(len(H)), np.argmax(score, axis=1)] = 1.
        elif read_mode in ('soft', 'mixture', 'blend'):
            p = _structural_softmax(self.temperature * score)
        else:
            raise ValueError('Read mode must be hard or soft')
        return p[0] if scalar else p

    def _read_weights(self, H, read_mode, recent_bias, reminders=None):
        p = self.route_probabilities(H, read_mode)
        bias = float(recent_bias)
        if not 0 <= bias <= 1:
            raise ValueError('Recent-access coefficient must be in [0,1]')
        if not bias:
            return p
        prior = np.zeros(self.banks)
        if reminders is not None and np.asarray(reminders).size:
            R, _ = _structural_rows(reminders)
            prior = self.route_probabilities(R, read_mode).mean(axis=0)
        elif self.active >= 0:
            prior[self.active] = 1.
        return (1-bias)*p + bias*prior if prior.sum() > 0 else p

    def output_view(self, H, read_mode='hard', recent_bias=0., reminders=None):
        H, scalar = _structural_rows(H)
        p = self._read_weights(H, read_mode, recent_bias, reminders)
        values = np.stack([self._bank_input(H, b) @ self._V[b]
                           for b in range(self.banks)], axis=1)
        result = np.einsum('nb,nbc->nc', p, values, optimize=False)
        return result[0] if scalar else result

    def output(self, H):
        return self.output_view(H, self.read_mode, self.recent_bias)

    def reminded_output(self, H, reminders):
        return self.output_view(H, self.read_mode, self.recent_bias, reminders)

    def observe(self, h, label):
        h = np.asarray(h, dtype=np.float64)
        q = unit_rows(h); target = target4(label)
        bank = int(np.argmax(self.centers @ q))
        local_h = self._bank_input(h, bank)
        prediction = local_h @ self._V[bank]
        error = target - prediction
        surprise = float(error @ error)
        active = np.abs(local_h) > 1e-12
        conflict = False
        if self.importance is not None:
            if self.release:
                # Each trace can decay autonomously between sensory events.
                self.importance *= self.recovery
                margin = float(np.max(np.delete(prediction, int(label))) - prediction[int(label)])
                conflict = margin > .30 and surprise > 1.25*self.surprise_ema[bank]
                if conflict:
                    self.importance[bank, active] *= .35
                    self.releases += 1
            used = np.take_along_axis(self.importance[bank], self.permutation[bank], axis=0)
            gain = 1. / (1. + self.strength*used)
        else:
            gain = np.ones((self.width, 1))
        change = delta_change(local_h, error, self.eta) * gain
        self._V[bank] += change
        norm2 = float(np.sum(change*change))
        # Fixed added teaching pathways broadcast the selected bank's error.
        # This intentionally increases total write size and is reported as such.
        if self.spillover:
            for other in range(self.banks):
                if other == bank:
                    continue
                hh = self._bank_input(h, other)
                extra = delta_change(hh, error, self.eta*self.spillover)
                self._V[other] += extra; norm2 += float(np.sum(extra*extra))
        achieved = float(np.linalg.norm(local_h @ change)) / max(float(np.linalg.norm(error)), 1e-12)
        if self.importance is not None:
            post_error = target - local_h @ self._V[bank]
            reliability = float(np.clip(1.-float(post_error @ post_error)/2., .05, 1.))
            credit = (np.outer(local_h*local_h, .10+target) if self.protection == 'pair'
                      else (.35*local_h*local_h)[:, None])
            self.importance[bank] += reliability*credit
            np.minimum(self.importance[bank], 1e6, out=self.importance[bank])
            self.surprise_ema[bank] = .95*self.surprise_ema[bank]+.05*surprise
        self.gain_sum += float(gain[active].mean()) if np.any(active) else 1.
        self.correction_sum += achieved
        self.empty_write_events += int(not np.any(active))
        self.counts[bank] += 1
        if self.routing == 'local':
            # Outcome-independent competition and local winning-cell prototype
            # adaptation; the rate uses only this cell's accumulated usage.
            rate = min(self.address_eta, 1./float(self.counts[bank]))
            self.centers[bank] = unit_rows((1.-rate)*self.centers[bank]+rate*q)
        self.route_changes += int(self.active >= 0 and bank != self.active)
        self.active = bank; self.t += 1; self.last_seen[bank] = self.t
        return {'extra_labeled_presentations': 0, 'update_norm': float(norm2**.5),
                'selected_bank': bank, 'release': int(conflict),
                'achieved_correction': achieved,
                'plasticity_gain': float(gain[active].mean()) if np.any(active) else 1.,
                'write_feature_energy': float(local_h @ local_h),
                'spillover_strength': self.spillover}

    def mutable_arrays(self):
        arrays = {'V': self._V, 'counts': self.counts, 'last_seen': self.last_seen}
        if self.routing == 'local':
            arrays['centers'] = self.centers
        if self.importance is not None:
            arrays.update(importance=self.importance, surprise_ema=self.surprise_ema)
        return arrays

    def fixed_arrays(self):
        arrays = {} if self.routing == 'local' else {'centers': self.centers}
        if self.feature_index is not None:
            arrays['feature_index'] = self.feature_index
        if self.permutation is not None:
            arrays['trace_routing'] = self.permutation
        return arrays

    def diagnostics(self):
        return {'bank_capacity': self.banks, 'used_banks': int(np.sum(self.counts > 0)),
                'route_changes_between_consecutive_inputs': self.route_changes,
                'readout_coefficients': int(self._V.size),
                'accessible_features_per_bank': self.width,
                'fixed_bytes': sum(a.nbytes for a in self.fixed_arrays().values()),
                'release_count': self.releases,
                'importance_mean': float(self.importance.mean()) if self.importance is not None else 0.,
                'mean_plasticity_gain': self.gain_sum/max(1, self.t),
                'mean_correction_fraction': self.correction_sum/max(1, self.t),
                'empty_eligible_write_events': self.empty_write_events}


class V24LegacyBankMemory(ContentBankMemory):
    """Archived bank-establishment rule with standardized pure evaluator views.

    Observe is inherited unchanged. Four/eight legacy banks retain the original
    sensory-novelty plus bank-local supervised-residual admission. This is an
    establishment-policy reference, not a clean isolated router-learning test.
    """
    def route_probabilities(self, H, read_mode='soft'):
        H, scalar = _structural_rows(H)
        q = unit_rows(H)
        p = np.zeros((len(H), self.banks))
        if self.routing == 'fixed':
            score = q @ self.fixed_address.T
            available = self.banks
        elif self.used:
            score = q @ self.centers[:self.used].T
            available = self.used
        else:
            return p[0] if scalar else p
        if read_mode in ('hard', 'winner'):
            p[np.arange(len(H)), np.argmax(score, axis=1)] = 1.
        elif read_mode in ('soft', 'mixture', 'blend'):
            p[:, :available] = _structural_softmax(self.temperature*score)
        else:
            raise ValueError('Read mode must be hard or soft')
        return p[0] if scalar else p

    def output_view(self, H, read_mode='soft', recent_bias=0., reminders=None):
        H, scalar = _structural_rows(H)
        p = self.route_probabilities(H, read_mode)
        prior = np.zeros(self.banks)
        if reminders is not None and np.asarray(reminders).size:
            R, _ = _structural_rows(reminders)
            prior = self.route_probabilities(R, read_mode).mean(axis=0)
        elif self.active >= 0:
            prior[self.active] = 1.
        if prior.sum() > 0:
            p = (1.-recent_bias)*p + recent_bias*prior[None, :]
        values = np.einsum('nd,bdc->nbc', H, self.V, optimize=False)
        result = np.sum(p[:, :, None]*values, axis=1)
        return result[0] if scalar else result

    def fixed_arrays(self):
        return {'fixed_address': self.fixed_address}


class V24AllocationMemory(MemoryBase):
    """Fixed-capacity prototype/value store with declared eviction policies.

    The original rule exactly retains v23's utility decay/reset behavior. Grace
    protects admissions for a fixed interval; if every slot is protected it
    refuses admission. Local recency decays an access trace and resets it on a
    hit; a fixed pairwise competition tree selects the least recent entry. In
    exact arithmetic its ordering is LRU, so these are implementation variants.
    All policies keep the prototype match/revision rule unchanged. No identity
    or evaluator history participates in admission, release, or replacement.
    """
    def __init__(self, d, seed, slots=128, policy='local_recency', grace=8,
                 threshold=.90, eta=.8, prototype_eta=.05, trace_decay=.99,
                 neighbors=1, **extra):
        super().__init__(d, seed)
        self.slots = int(slots)
        self.policy = {'recency': 'lru', 'local': 'local_recency',
                       'no_recycling': 'none', 'frequency': 'original'}.get(policy, policy)
        if self.slots < 1 or self.policy not in ('original', 'none', 'lru', 'grace', 'local_recency'):
            raise ValueError('Unknown allocation capacity/policy')
        self.grace = max(0, int(grace)); self.threshold = float(threshold)
        self.eta = float(eta); self.prototype_eta = float(prototype_eta)
        self.trace_decay = float(trace_decay); self.neighbors = max(1, int(neighbors))
        if not 0 < self.trace_decay < 1:
            raise ValueError('Local recency decay must lie strictly between zero and one')
        self.centers = np.zeros((self.slots, self.d))
        self.values = np.zeros((self.slots, 4))
        self.counts = np.zeros(self.slots, dtype=np.int64)
        self.born = np.full(self.slots, -1, dtype=np.int64)
        self.last_hit = np.full(self.slots, -1, dtype=np.int64)
        self.generation = np.zeros(self.slots, dtype=np.int64)
        self.utility = np.zeros(self.slots)
        self.recency = np.zeros(self.slots) if self.policy == 'local_recency' else None
        self.used = 0; self.allocations = 0; self.replacements = 0
        self.rejected = 0; self.revisions = 0; self.first_full_event = -1
        self.victim_age_sum = 0.; self.victim_hit_age_sum = 0.
        self.grace_refusals = 0

    def selected_slots(self, H):
        H, scalar = _structural_rows(H)
        if not self.used:
            result = np.full(len(H), -1, dtype=np.int64)
        else:
            result = np.argmax(unit_rows(H) @ self.centers[:self.used].T, axis=1)
        return int(result[0]) if scalar else result

    def output(self, H):
        H, scalar = _structural_rows(H)
        result = np.zeros((len(H), 4))
        if self.used:
            similarities = unit_rows(H) @ self.centers[:self.used].T
            if self.neighbors == 1:
                result = self.values[np.argmax(similarities, axis=1)].copy()
            else:
                n = min(self.used, self.neighbors)
                index = np.argsort(similarities, axis=1, kind='stable')[:, -n:]
                p = _structural_softmax(16*np.take_along_axis(similarities, index, axis=1))
                result = np.sum(p[:, :, None]*self.values[index], axis=1)
        return result[0] if scalar else result

    def _victim(self, event):
        if self.policy == 'none':
            return -1
        if self.policy == 'original':
            return int(np.argmin(self.utility[:self.used]))
        if self.policy == 'lru':
            return int(np.argmin(self.last_hit[:self.used]))
        if self.policy == 'grace':
            eligible = np.flatnonzero(event-self.born[:self.used] >= self.grace)
            if not len(eligible):
                self.grace_refusals += 1
                return -1
            return int(eligible[np.argmin(self.utility[eligible])])
        return v24_competition(-self.recency[:self.used])

    def observe(self, h, label):
        h = unit_rows(np.asarray(h, dtype=np.float64)); target = target4(label)
        event = self.t+1
        self.utility *= .999
        if self.recency is not None:
            self.recency *= self.trace_decay
        best, similarity = -1, -1.
        if self.used:
            similarities = self.centers[:self.used] @ h
            best = (v24_competition(similarities) if self.policy == 'local_recency'
                    else int(np.argmax(similarities)))
            similarity = float(similarities[best])
        novel = similarity < self.threshold
        replacing = False; victim_age = -1; victim_hit_age = -1
        pre_used = self.used
        if novel:
            if self.used < self.slots:
                best = self.used; self.used += 1; self.allocations += 1
            else:
                best = self._victim(event)
                if best < 0:
                    self.rejected += 1; self.t = event
                    return {'extra_labeled_presentations': 0, 'update_norm': 0.,
                            'selected_slot': -1, 'allocated': 0, 'recycled': 0,
                            'rejected': 1, 'capacity_rejection': 1,
                            'match_similarity': similarity, 'victim_age': -1,
                            'victim_last_hit_age': -1, 'novel': 1,
                            'first_full_event': self.first_full_event}
                replacing = True; self.replacements += 1
                victim_age = event-int(self.born[best]); victim_hit_age = event-int(self.last_hit[best])
                self.victim_age_sum += victim_age; self.victim_hit_age_sum += victim_hit_age
            old_value = self.values[best].copy()
            self.centers[best] = h; self.values[best] = target
            self.generation[best] += 1
            self.counts[best] = 1; self.utility[best] = 1.; self.born[best] = event
        else:
            old_value = self.values[best].copy()
            self.revisions += int(np.argmax(old_value) != int(label))
            self.centers[best] = unit_rows((1.-self.prototype_eta)*self.centers[best]+self.prototype_eta*h)
            self.values[best] += self.eta*(target-self.values[best])
            self.counts[best] += 1; self.utility[best] += 1.
        self.last_hit[best] = event
        if self.recency is not None:
            self.recency[best] = 1.
        self.t = event
        if self.used == self.slots and self.first_full_event < 0:
            self.first_full_event = event
        return {'extra_labeled_presentations': 0,
                'update_norm': float(np.linalg.norm(self.values[best]-old_value)),
                'selected_slot': best, 'allocated': int(novel and not replacing),
                'recycled': int(replacing), 'rejected': 0, 'capacity_rejection': 0,
                'match_similarity': similarity, 'victim_age': victim_age,
                'victim_last_hit_age': victim_hit_age, 'novel': int(novel),
                'became_full': int(pre_used < self.slots and self.used == self.slots),
                'first_full_event': self.first_full_event}

    def begin_teaching(self, h):
        """Sensory admission for a separately declared delayed-teaching circuit.

        This event advances the sensory clock and prototype state without a
        target. New entries have zero values; existing values are untouched.
        The slot-generation tag is bounded per-slot learner state, not a cue
        identity. A pending trace can refuse a write after that slot is reused.
        Full supervised observe retains its original v23 numerical rule.
        """
        h = unit_rows(np.asarray(h, dtype=np.float64)); event = self.t+1
        self.utility *= .999
        if self.recency is not None:
            self.recency *= self.trace_decay
        best, similarity = -1, -1.
        if self.used:
            similarities = self.centers[:self.used] @ h
            best = (v24_competition(similarities) if self.policy == 'local_recency'
                    else int(np.argmax(similarities)))
            similarity = float(similarities[best])
        novel = similarity < self.threshold
        replacing = False; victim_age = -1; victim_hit_age = -1
        previous_used = self.used
        if novel:
            if self.used < self.slots:
                best = self.used; self.used += 1; self.allocations += 1
            else:
                best = self._victim(event)
                if best < 0:
                    self.rejected += 1; self.t = event
                    return {'selected_slot': -1, 'generation': -1, 'allocated': 0,
                            'recycled': 0, 'rejected': 1, 'update_norm': 0.,
                            'extra_labeled_presentations': 0,
                            'first_full_event': self.first_full_event}
                replacing = True; self.replacements += 1
                victim_age = event-int(self.born[best]); victim_hit_age = event-int(self.last_hit[best])
                self.victim_age_sum += victim_age; self.victim_hit_age_sum += victim_hit_age
            self.centers[best] = h; self.values[best] = 0.
            self.generation[best] += 1
            self.counts[best] = 1; self.utility[best] = 1.; self.born[best] = event
        else:
            self.centers[best] = unit_rows((1.-self.prototype_eta)*self.centers[best]+self.prototype_eta*h)
            self.counts[best] += 1; self.utility[best] += 1.
        self.last_hit[best] = event
        if self.recency is not None:
            self.recency[best] = 1.
        self.t = event
        if self.used == self.slots and self.first_full_event < 0:
            self.first_full_event = event
        return {'selected_slot': best, 'generation': int(self.generation[best]),
                'allocated': int(novel and not replacing), 'recycled': int(replacing),
                'rejected': 0, 'novel': int(novel), 'match_similarity': similarity,
                'victim_age': victim_age, 'victim_last_hit_age': victim_hit_age,
                'became_full': int(previous_used < self.slots and self.used == self.slots),
                'first_full_event': self.first_full_event,
                'extra_labeled_presentations': 0, 'update_norm': 0.}

    def mutable_arrays(self):
        arrays = {'centers': self.centers, 'values': self.values,
                  'counts': self.counts, 'born': self.born,
                  'last_hit': self.last_hit, 'utility': self.utility,
                  'generation': self.generation}
        if self.recency is not None:
            arrays['recency'] = self.recency
        return arrays

    def fixed_arrays(self):
        return {}

    def diagnostics(self):
        return {'used_slots': self.used, 'slot_capacity': self.slots,
                'total_allocations': self.allocations, 'slot_replacements': self.replacements,
                'capacity_rejections': self.rejected, 'grace_refusals': self.grace_refusals,
                'conflicting_label_revisions': self.revisions,
                'first_full_event': self.first_full_event,
                'mean_victim_age': self.victim_age_sum/max(1, self.replacements),
                'mean_victim_last_hit_age': self.victim_hit_age_sum/max(1, self.replacements),
                'local_trace_zero_count': int(np.sum(self.recency[:self.used] == 0))
                                           if self.recency is not None else 0}


class V24LocalTurnoverMemory(MemoryBase):
    """Winner-local address learning with optional bounded contact turnover.

    Each unit has a fixed number of afferent contacts, plus a local output
    readout. A winning unit updates its contacted weights from its own input
    and activity. No V-transpose or hidden derivative is used. Turnover copies
    a bounded sensory sample into the selected low-usage unit, resets its local
    output state, and retains the exact contact count. Selection uses a declared
    usage competition. Both arms learn their surviving addresses identically.
    """
    def __init__(self, d, seed, width=96, k=24, replace=False, interval=48,
                 eta=.6, address_eta=.015, contacts=16, units=None, active=None,
                 recycle=None, **extra):
        super().__init__(d, seed)
        self.width = int(width if units is None else units)
        self.k = min(int(k if active is None else active), self.width)
        self.replace = bool(replace if recycle is None else recycle)
        self.interval = max(1, int(interval)); self.eta = float(eta)
        self.address_eta = float(address_eta); self.contacts = min(self.d, max(1, int(contacts)))
        self.indices = np.stack([self.rng.choice(self.d, self.contacts, replace=False)
                                 for _ in range(self.width)]).astype(np.int64)
        self.weights = unit_rows(self.rng.normal(size=(self.width, self.contacts)))
        self.V = np.zeros((self.width, 4))
        self.utility = np.zeros(self.width); self.age = np.zeros(self.width, dtype=np.int64)
        self.usage = np.zeros(self.width, dtype=np.int64)
        self.replacements = 0

    def _encode(self, H):
        H, scalar = _structural_rows(H)
        q = unit_rows(H)
        local = q[:, self.indices]
        activation = np.maximum(np.sum(local*self.weights[None, :, :], axis=2), 0.)
        indices = np.argsort(activation, axis=1, kind='stable')[:, -self.k:]
        mask = np.zeros_like(activation, dtype=bool)
        np.put_along_axis(mask, indices, True, axis=1)
        code = unit_rows(activation*mask)
        return code, local, mask, scalar

    def output(self, H):
        code, _, _, scalar = self._encode(H)
        result = code @ self.V
        return result[0] if scalar else result

    def observe(self, h, label):
        h = np.asarray(h, dtype=np.float64)
        code, local, mask, _ = self._encode(h)
        u = code[0]; winners = mask[0] & (u > 0)
        before = self.V.copy()
        self.V += delta_change(u, target4(label)-u @ self.V, self.eta)
        if np.any(winners):
            rate = self.address_eta*u[winners, None]
            self.weights[winners] = unit_rows((1.-rate)*self.weights[winners]+rate*local[0, winners])
        self.utility *= .99; self.utility += .01*np.abs(u)
        self.age += 1; self.usage[winners] += 1; self.t += 1
        replaced = 0
        if self.replace and self.t % self.interval == 0:
            eligible = np.flatnonzero(self.age >= self.interval)
            if len(eligible):
                j = int(eligible[v24_competition(-self.utility[eligible])])
                # A fixed number of locally contacted sensory proposals. The
                # random generator implements an explicitly prescribed proposal
                # source; no output label selects the replacement connections.
                self.indices[j] = self.rng.choice(self.d, self.contacts, replace=False)
                proposal = h[self.indices[j]].copy()
                if np.linalg.norm(proposal) <= 1e-12:
                    proposal = self.rng.normal(size=self.contacts)
                self.weights[j] = unit_rows(proposal)
                self.V[j] = 0.; self.utility[j] = 0.; self.age[j] = 0
                self.usage[j] = 0; self.replacements += 1; replaced = 1
        return {'extra_labeled_presentations': 0,
                'update_norm': float(np.linalg.norm(self.V-before)),
                'replaced_units': replaced, 'active_units': int(winners.sum())}

    def mutable_arrays(self):
        arrays = {'weights': self.weights, 'V': self.V, 'utility': self.utility,
                  'age': self.age, 'usage': self.usage}
        if self.replace:
            arrays['indices'] = self.indices
        return arrays

    def fixed_arrays(self):
        return {} if self.replace else {'indices': self.indices}

    def diagnostics(self):
        return {'allocated_units': self.width, 'replacements': self.replacements,
                'contacts_per_unit': self.contacts, 'contact_count': int(self.indices.size),
                'currently_unused_units': int(np.sum(self.usage == 0)),
                'mean_utility': float(self.utility.mean()),
                'fixed_bytes': sum(x.nbytes for x in self.fixed_arrays().values())}


class V24ExpansionMemory(MemoryBase):
    """Non-gated fixed sparse nonlinear expansion with local delta readout.

    Every expansion unit may be active; there is no bank-routing variable.
    Sparse addresses are stored as indices and values, and counted at their
    actual dtypes. Width is chosen mechanically for a byte-budget comparison.
    """
    def __init__(self, d, seed, width=96, contacts=8, eta=.6, **extra):
        super().__init__(d, seed)
        self.width = int(width); self.contacts = min(self.d, max(1, int(contacts)))
        self.eta = float(eta)
        if self.width < 1:
            raise ValueError('Expansion must have positive width')
        self.indices = np.stack([self.rng.choice(self.d, self.contacts, replace=False)
                                 for _ in range(self.width)]).astype(np.int64)
        self.address = unit_rows(self.rng.normal(size=(self.width, self.contacts)))
        self.V = np.zeros((self.width, 4))

    def _encode(self, H):
        H, scalar = _structural_rows(H)
        activation = np.maximum(np.sum(H[:, self.indices]*self.address[None, :, :], axis=2), 0.)
        return unit_rows(activation), scalar

    def output(self, H):
        code, scalar = self._encode(H)
        out = code @ self.V
        return out[0] if scalar else out

    def observe(self, h, label):
        code, _ = self._encode(h); u = code[0]
        change = delta_change(u, target4(label)-u @ self.V, self.eta)
        self.V += change; self.t += 1
        return {'extra_labeled_presentations': 0,
                'update_norm': float(np.linalg.norm(change)),
                'active_units': int(np.sum(u > 0))}

    def mutable_arrays(self):
        return {'V': self.V}

    def fixed_arrays(self):
        return {'indices': self.indices, 'address': self.address}

    def diagnostics(self):
        return {'expansion_width': self.width, 'contacts_per_unit': self.contacts,
                'fixed_bytes': sum(a.nbytes for a in self.fixed_arrays().values())}


def v24_budget_memory(d, seed, target_banks=8, alternative='expansion', byte_cost=None):
    """Fit functional alternatives to the fixed-bank persistent memory budget.

    The runner may pass its own total persistent-state cost function, including
    scalar and RNG state. Without it the explicit default is NumPy array bytes.
    Width/slots are derived before observing scores; unused padding is absent.
    This does not claim equality of expressive capacity or working memory.
    """
    cost = v24_memory_array_bytes if byte_cost is None else byte_cost
    target = int(cost(V24BankMemory(d, seed, banks=target_banks, routing='fixed')))
    if alternative == 'expansion':
        factory = lambda n: V24ExpansionMemory(d, seed, width=n)
    elif alternative == 'allocation':
        factory = lambda n: V24AllocationMemory(d, seed, slots=n, policy='lru')
    elif alternative == 'replay':
        factory = lambda n: ReplayMemory(d, seed, capacity=n, policy='recent')
    else:
        raise ValueError('Unknown budget alternative: ' + str(alternative))
    one, two = factory(1), factory(2)
    per = int(cost(two))-int(cost(one))
    fixed = int(cost(one))-per
    if per <= 0 or int(cost(one)) > target:
        raise ValueError('Target budget cannot support the minimum functional alternative')
    count = max(1, (target-fixed)//per)
    model = factory(count)
    # Cost functions that include discrete metadata may not be perfectly affine.
    # Correct to the largest fitting size without consulting outcomes.
    while count > 1 and int(cost(model)) > target:
        count -= 1; model = factory(count)
    while int(cost(factory(count+1))) <= target:
        count += 1; model = factory(count)
    actual = int(cost(model))
    if actual > target:
        raise ValueError('Target budget cannot support the minimum functional alternative')
    model.v24_target_array_bytes = int(target)
    model.v24_unused_array_bytes = int(target-actual)
    model.v24_budget_target_banks = int(target_banks)
    model.v24_budget_basis = 'persistent_arrays' if byte_cost is None else 'runner_persistent_state'
    return model


def v24_memory_checks():
    """Small arithmetic/causality gates; these are not scientific experiments."""
    import copy
    rng = np.random.default_rng(4021); d = 16
    H = unit_rows(rng.normal(size=(30, d)))
    checks = {}
    a = V24BankMemory(d, 11, banks=4, routing='fixed')
    b = V24BankMemory(d, 11, banks=4, routing='local')
    assert np.array_equal(a.centers, b.centers)
    assert np.array_equal(a.route_probabilities(H), b.route_probabilities(H))
    checks['fixed_and_local_start_with_identical_routes'] = True
    views = [V24BankMemory(d, 11, banks=4, routing='local', protection='pair',
                          release=True, read_mode=mode, recent_bias=bias)
             for mode in ('hard', 'soft') for bias in (0., .35)]
    for i, h in enumerate(H):
        for model in views:
            model.output(h); model.observe(h, i % 4)
    for model in views[1:]:
        for key, array in views[0].mutable_arrays().items():
            assert np.array_equal(array, model.mutable_arrays()[key])
    checks['pure_read_views_preserve_identical_learning_trajectories'] = True
    one = V24BankMemory(d, 12, banks=1)
    ordinary = PlainMemory(d, 12, eta=.6)
    for i, h in enumerate(H):
        one.observe(h, i % 4); ordinary.observe(h, i % 4)
    assert np.allclose(one.V[0], ordinary.V, atol=1e-14)
    checks['one_bank_matches_ordinary_delta_learning'] = True
    isolated = V24BankMemory(d, 13, banks=4)
    j = isolated.observe(H[0], 0)['selected_bank']
    assert np.all(isolated.V[np.arange(4) != j] == 0)
    spill = V24BankMemory(d, 13, banks=4, spillover=.1)
    spill.observe(H[0], 0)
    assert np.allclose(spill.V[j], isolated.V[j]) and np.all(np.linalg.norm(spill.V, axis=(1, 2)) > 0)
    checks['spillover_changes_other_banks_at_identical_selected_correction'] = True
    compact = V24BankMemory(d, 2, banks=4, coefficient_budget=4*d)
    assert compact._V.size == 4*d and compact.V.shape == (4, d, 4)
    compact.observe(H[0], 1)
    for bank in range(4):
        excluded = np.setdiff1d(np.arange(d), compact.feature_index[bank])
        assert np.all(compact.V[bank, excluded] == 0)
    checks['coefficient_control_stores_only_budgeted_weights'] = True
    # The archived allocator must reproduce stored addresses/values and victims.
    original = BoundedAllocationMemory(d, 77, slots=3)
    revised = V24AllocationMemory(d, 77, slots=3, policy='original')
    for i, h in enumerate(H):
        old_info = original.observe(h, i % 4); new_info = revised.observe(h, i % 4)
        assert np.allclose(original.values, revised.values)
        assert np.allclose(original.centers, revised.centers)
        assert np.allclose(original.utility, revised.utility)
        assert old_info.get('recycled', 0) == new_info['recycled']
    checks['original_eviction_and_prototype_rule_match_v23'] = True
    legacy = ContentBankMemory(d, 19, banks=4, routing='inferred')
    wrapped = V24LegacyBankMemory(d, 19, banks=4, routing='inferred')
    for i, h in enumerate(H):
        legacy.observe(h, i % 4); wrapped.observe(h, i % 4)
        assert np.array_equal(legacy.V, wrapped.V)
        assert np.allclose(legacy.output(H[:3]), wrapped.output_view(H[:3], 'soft', .35))
    checks['legacy_bank_wrapper_preserves_archived_training_and_expression'] = True
    delayed = V24AllocationMemory(d, 4, slots=1, policy='local_recency')
    info1 = delayed.begin_teaching(H[0])
    assert np.all(delayed.values == 0) and info1['generation'] == 1
    delayed.values[0, 2] = .75
    again = delayed.begin_teaching(H[0])
    assert again['generation'] == 1 and delayed.values[0, 2] == .75
    changed = delayed.begin_teaching(H[1])
    assert changed['generation'] == 2 and np.all(delayed.values == 0)
    checks['sensory_admission_is_label_free_and_slot_reuse_changes_generation'] = True
    lru = V24AllocationMemory(d, 8, slots=3, policy='lru')
    local = V24AllocationMemory(d, 8, slots=3, policy='local_recency')
    for i, h in enumerate(H):
        li = lru.observe(h, i % 4); lo = local.observe(h, i % 4)
        assert li['selected_slot'] == lo['selected_slot']
        assert np.allclose(lru.values, local.values)
    checks['recency_and_local_trace_have_same_ranking_before_precision_loss'] = True
    grace = V24AllocationMemory(d, 8, slots=2, policy='grace', grace=20)
    for i in range(4):
        last = grace.observe(H[i], i)
    assert grace.used == 2 and last['rejected'] == 1 and grace.grace_refusals == 2
    checks['all_protected_grace_fallback_refuses_without_capacity_growth'] = True
    # Demonstrate repair in the isolated saturation sequence which exposed v23.
    old = V24AllocationMemory(d, 8, slots=2, policy='original')
    repair = V24AllocationMemory(d, 8, slots=2, policy='lru')
    basis = np.eye(d)
    sequence = [0]*6+[1]*6+[2, 3]*4
    for j in sequence:
        old.observe(basis[j], j); repair.observe(basis[j], j)
    assert np.max(old.centers @ basis[:2].T) > .99
    assert np.max(repair.centers @ basis[2]) > .99 and np.max(repair.centers @ basis[3]) > .99
    checks['recency_repair_accepts_recurring_new_cues_after_saturation'] = True
    retained = V24LocalTurnoverMemory(d, 1, width=16, k=4, interval=4, contacts=4)
    recycled = V24LocalTurnoverMemory(d, 1, width=16, k=4, interval=4, contacts=4, replace=True)
    for i, h in enumerate(H[:3]):
        retained.observe(h, i % 4); recycled.observe(h, i % 4)
    assert np.array_equal(retained.weights, recycled.weights)
    for i, h in enumerate(H[3:]):
        recycled.observe(h, i % 4)
    assert recycled.replacements > 0 and recycled.indices.size == 16*4
    checks['turnover_preserves_contact_bound_without_hidden_gradients'] = True
    alternate_labels = V24LocalTurnoverMemory(d, 1, width=16, k=4, interval=4, contacts=4)
    original_labels = V24LocalTurnoverMemory(d, 1, width=16, k=4, interval=4, contacts=4)
    for i, h in enumerate(H):
        alternate_labels.observe(h, (i+1) % 4); original_labels.observe(h, i % 4)
    assert np.array_equal(alternate_labels.weights, original_labels.weights)
    assert np.array_equal(alternate_labels.utility, original_labels.utility)
    checks['local_addresses_and_turnover_utility_do_not_depend_on_output_targets'] = True
    for target in (4, 8):
        for alt in ('expansion', 'allocation', 'replay'):
            model = v24_budget_memory(d, 1, target, alt)
            assert v24_memory_array_bytes(model) <= model.v24_target_array_bytes
    checks['functional_alternatives_fit_persistent_array_budget'] = True
    models = views+[a, b, compact, isolated, spill, original, revised, lru, local,
                    grace, retained, recycled, V24ExpansionMemory(d, 3, width=24)]
    for model in models:
        state = model.state_digest()
        assert model.output(H[:3]).shape == (3, 4)
        assert model.output(H[0]).shape == (4,)
        assert state == model.state_digest()
        assert all(np.all(np.isfinite(x)) for x in model.mutable_arrays().values())
        assert np.all(np.isfinite(model.output(H)))
    checks['all_predictions_pure_shapes_and_states_finite'] = True
    return checks


# === V24 manifest ===
"""V24 predeclared experiment roster; stdlib only at import time.

This fragment is appended to the self-contained V23 source.  The manifest is
data, not a search over scores: every default choice below precedes execution.
Expression-only bank views reuse storage only in the supervised selected-bank
training panel.  Reward conditions are independent trajectories.
"""
import copy as _v24_copy


V24_ENCODER_LABELS = {
    'E0': 'Archived clean-target backprop development; frozen historical control',
    'E1': 'Observed-target backprop development; frozen development control',
    'E2': 'Prescribed circuitry without weight development; calibration counted',
    'E3': 'Local sensory afferent development and homeostasis; no hidden backprop',
}
V24_PANELS = ('frozen', 'live', 'circuit', 'noise', 'structured', 'teaching',
              'confirmation')
V24_DEFAULT_CONFIRMATION_SEEDS = (101, 127, 149, 173, 197, 223)
V24_ANCHOR_NAMES = (
    'ordinary', 'bank4_fixed', 'bank8_fixed', 'bank8_local',
    'bank8_local_neuron_release', 'allocate128_original',
    'allocate128_lru', 'allocate128_local_recency', 'fast_slow3',
)


def v24_memory_specs():
    """101 pre-deduplication specifications, including all 40 family rows.

    The one-bank implementation is retained as a numerical-equivalence check;
    ordinary and bank1 do not silently share costs or traces.  Its scalar
    routing duplicates are omitted.  Actual budget alternatives may coincide
    in width/slots, which is reported rather than disguised as replication.
    """
    removed = {
        'historical_ridge', 'projection16_strict', 'projection48_strict',
        'projection16_leaky', 'quantized_allocate128', 'adapter_retained',
        'adapter_recycled', 'allocate32', 'allocate128',
        'allocate32_no_recycle', 'silent_fixed', 'silent_inferred',
        'context_fixed', 'context_inferred',
    }
    specs = []
    abstract = {'v21_input', 'v22_joint', 'v22_scrambled',
                'pair_importance_shuffled', 'joint_plus_importance'}
    for old in arm_manifest():
        if old['name'] in removed:
            continue
        s = _v24_copy.deepcopy(old)
        s.update(lineage=[old['name']], panels=['frozen'],
                 locality=('software_anatomy_or_shuffle_reference' if
                           old['name'] in abstract else
                           'no_hidden_backprop_with_declared_circuit_abstractions'),
                 parent=None if old['name'] == 'ordinary' else 'ordinary',
                 contrast='memory_mechanism', gate=False)
        if s['name'] == 'ordinary_slow':
            s['contrast'] = 'learning_rate'
        if s['name'] == 'repeat_current':
            s['contrast'] = 'extra_current_presentations'
        if s['name'] == 'pair_importance_strong':
            s['parent'], s['contrast'] = 'pair_importance', 'protection_strength'
        if s['name'] == 'importance_surprise_release':
            s['parent'], s['contrast'] = 'pair_importance', 'local_error_release'
        if s['name'] == 'pair_importance_shuffled':
            s['parent'], s['contrast'] = 'pair_importance', 'trace_address_shuffle'
        if s['name'] == 'bayes_reopening':
            s['parent'], s['contrast'] = 'bayes_stable', 'uncertainty_reopening'
        if s['family'] == 'replay_reference':
            s['note'] += (' Exact bounded buffer is a software reactivation '
                          'reference; extra presentations and stored addresses count.')
        specs.append(s)

    def add(name, family, kind, params=None, feature='sparse', note='',
            parent='ordinary', contrast='memory_mechanism', locality='local_rule',
            lineage=None, gate=False, **extra):
        s = dict(name=name, family=family, kind=kind, params=params or {},
                 feature=feature, note=note, locality=locality, panels=['frozen'],
                 parent=parent, contrast=contrast, lineage=lineage or [], gate=gate)
        s.update(extra)
        specs.append(s)

    for replace in (False, True):
        add('local_turnover_' + ('recycled' if replace else 'retained'),
            'local_capacity_turnover', 'turnover24',
            dict(width=96, k=24, replace=replace, interval=48), feature='dense',
            parent='local_turnover_retained' if replace else 'ordinary',
            contrast='local_address_replacement' if replace else 'local_address_rule',
            lineage=['adapter_' + ('recycled' if replace else 'retained')],
            note='New local competitive address rule replaces hidden-gradient adapter; '
                 'old adapter scores are not inherited.')
    assert len(specs) == 40, 'Surviving family roster must have 40 rows.'

    bankcommon = dict(eta=.6, read_mode='hard', recent_bias=0.)
    views = [dict(read_mode=r, recent_bias=b)
             for r in ('hard', 'soft') for b in (0., .35)]
    add('bank1', 'bank_factorial', 'bank24', dict(bankcommon, banks=1, routing='fixed'),
        contrast='single_bank_implementation_equivalence',
        note='One-bank routing/read/bias duplicates collapsed. Check numerical '
             'equivalence to ordinary; actual router storage remains charged.',
        equivalence_candidate='ordinary', expression_views=[dict(read_mode='hard', recent_bias=0.)])
    for banks in (2, 4, 8):
        for routing in ('fixed', 'local'):
            name = 'bank%d_%s' % (banks, routing)
            parent = ('bank%d_fixed' % banks if routing == 'local' else
                      ('bank1' if banks == 2 else 'bank%d_fixed' % (banks // 2)))
            add(name, 'bank_factorial', 'bank24',
                dict(bankcommon, banks=banks, routing=routing), parent=parent,
                contrast='sensory_router_learning' if routing == 'local' else 'bank_count',
                expression_views=_v24_copy.deepcopy(views),
                note='All banks start available with matched prototype draws. '
                     'Selected-bank residual and sensory router updates do not depend '
                     'on evaluation read mode or recent bias.')

    for banks in (4, 8):
        add('bank%d_legacy_inferred' % banks, 'legacy_bank_admission',
            'context_gating' if banks == 4 else 'silent_memory',
            dict(banks=banks, routing='inferred'),
            parent='bank%d_local' % banks, contrast='initialization_and_admission',
            locality='archived_outcome_gated_admission_reference',
            lineage=['context_inferred' if banks == 4 else 'silent_inferred'],
            note='Archived ContentBankMemory numerical rule; original expression bias '
                 'is declared by the factory. Initialization and admission differ '
                 'jointly from the common-initialization sensory router.')

    for banks in (4, 8):
        for routing in ('fixed', 'local'):
            base = 'bank%d_%s' % (banks, routing)
            for spill in (.1, .5):
                add(base + '_spill%d' % round(100 * spill), 'bank_write_isolation',
                    'bank24', dict(bankcommon, banks=banks, routing=routing, spillover=spill),
                    parent=base, contrast='nonselected_bank_teaching',
                    expression_views=_v24_copy.deepcopy(views),
                    note='Identical memory arrays and input routing; selected bank '
                         'keeps ordinary correction; other banks receive fixed spillover.')
            add(base + '_coeff384', 'bank_coefficient_budget', 'bank24',
                dict(bankcommon, banks=banks, routing=routing, coefficient_budget=384),
                parent=base, contrast='readout_coefficient_budget',
                expression_views=_v24_copy.deepcopy(views),
                note='384 actual plastic coefficients across all banks; prespecified '
                     'feature subsets change access. Router costs remain additional.')
    for banks in (4, 8):
        for alternative in ('expansion', 'allocation', 'replay'):
            add('budget%d_%s' % (banks, alternative), 'total_memory_budget', 'budget24',
                dict(target_banks=banks, alternative=alternative),
                parent='bank%d_fixed' % banks, contrast='functional_equal_byte_alternative',
                locality='local_readout_with_declared_storage_and_competition',
                note='Width/slots derived mechanically from actual bank fixed+mutable '
                     'array bytes before seeing data; includes addresses, counters '
                     'and traces. Report target, used and unused bytes.')
    assert len(specs) == 67

    conditions = (
        ('slow', dict(eta=.15), 'learning_rate'),
        ('neuron', dict(protection='neuron', strength=3.), 'neuron_protection'),
        ('neuron_release', dict(protection='neuron', strength=3., release=True,
                               recovery=.997), 'neuron_recovery_and_release'),
        ('pair', dict(protection='pair', strength=3.), 'pair_protection'),
        ('pair_release', dict(protection='pair', strength=3., release=True,
                             recovery=.997), 'pair_recovery_and_release'),
        ('shuffled', dict(protection='pair', strength=3., scramble=True),
         'trace_address_shuffle'),
    )
    for banks in (4, 8):
        for routing in ('fixed', 'local'):
            base = 'bank%d_%s' % (banks, routing)
            for suffix, p, contrast in conditions:
                params = dict(bankcommon, banks=banks, routing=routing)
                params.update(p)
                parent = (base + '_neuron' if suffix == 'neuron_release' else
                          base + '_pair' if suffix in ('pair_release', 'shuffled') else base)
                add(base + '_' + suffix, 'within_bank_protection', 'bank24', params,
                    parent=parent, contrast=contrast, gate=suffix in ('neuron', 'pair'),
                    expression_views=_v24_copy.deepcopy(views),
                    locality='software_trace_shuffle_reference' if suffix == 'shuffled' else 'local_rule',
                    note='Suppression is retained in achieved correction. Release '
                         'uses selected-bank error and active traces, never phase IDs.',
                    additional_parent=base)
    assert len(specs) == 91
    for slots in (32, 128):
        for policy in ('original', 'none', 'lru', 'grace', 'local_recency'):
            name = 'allocate%d_%s' % (slots, policy)
            parent = 'ordinary' if policy == 'original' else 'allocate%d_original' % slots
            add(name, 'bounded_allocation_repair', 'allocation24',
                dict(slots=slots, policy=policy, threshold=.9, eta=.8,
                     prototype_eta=.05, grace=8), parent=parent, contrast='eviction_policy',
                locality=('exact_recency_software_reference' if policy == 'lru' else
                          'bounded_competition_circuit_candidate'),
                lineage=(['allocate%d' % slots] if policy == 'original' else
                         ['allocate32_no_recycle'] if policy == 'none' and slots == 32 else []),
                note='Fixed capacity and matching prototypes/values. Grace is eight '
                     'external events, with declared bounded fallback. Saturation '
                     'and victim labels are evaluator diagnostics only.',
                equivalence_candidate=('allocate%d_lru' % slots if policy == 'local_recency' else None))
    assert len(specs) == 101
    assert len({s['name'] for s in specs}) == len(specs)
    return specs


def _v24_selected(value, default):
    if value is None:
        return list(default)
    if isinstance(value, str):
        value = value.replace(',', ' ').split()
    return list(value)


def v24_jobs(cfg):
    """Generate stable jobs, explicit paired parents and fixed confirmation.

    Selected panels automatically retain their available control dependencies.
    An explicit ``arms`` list is a strict memory-name filter: control expansion
    may add another panel/variant of a requested memory, but never an unrequested
    memory. Comparisons to excluded parents retain their IDs and are explicitly
    unpaired in reporting. No results enter this function. ``smoke`` alters the
    environment elsewhere; it does not silently remove mechanism families here.
    """
    specs = {s['name']: s for s in v24_memory_specs()}
    arms_arg = getattr(cfg, 'arms', None)
    requested_arms = None
    if arms_arg is not None:
        arms = _v24_selected(arms_arg, ())
        if not arms:
            raise ValueError('--arms requires at least one exact memory name.')
        if len(set(arms)) != len(arms):
            raise ValueError('--arms contains duplicate memory names.')
        unknown_arms = set(arms) - set(specs)
        if unknown_arms:
            raise ValueError('Unknown V24 memory name(s) in --arms: ' +
                             ', '.join(sorted(unknown_arms)) +
                             '. Use --list-arms without --arms to list valid names.')
        requested_arms = set(arms)
    seeds = [int(s) for s in _v24_selected(getattr(cfg, 'seeds', None),
                                        (17, 29, 43, 59, 71, 89))]
    encoders = _v24_selected(getattr(cfg, 'encoders', None), ('E0', 'E1', 'E2', 'E3'))
    if any(e not in V24_ENCODER_LABELS for e in encoders):
        raise ValueError('Encoders must be E0, E1, E2 or E3.')
    panels = _v24_selected(getattr(cfg, 'panels', None), ('all',))
    if 'all' in panels:
        panels = list(V24_PANELS)
    unknown = set(panels) - set(V24_PANELS)
    if unknown:
        raise ValueError('Unknown V24 panels: ' + ', '.join(sorted(unknown)))
    jobs, signatures = [], {}

    def add(panel, seed, encoder, name, variant=None, order='original',
            task='association', teaching=None, parent=None, contrast=None,
            comparison_factor=None, memory_overrides=None, **extra):
        memory = _v24_copy.deepcopy(specs[name])
        if memory_overrides:
            memory['params'].update(memory_overrides)
        job = dict(panel=panel, seed=int(seed), encoder=encoder, memory=memory,
                   encoder_variant=variant or {}, order=order, task=task,
                   teaching=teaching, encoder_description=V24_ENCODER_LABELS[encoder],
                   contrast=contrast or memory.get('contrast'),
                   comparison_factor=comparison_factor or [],
                   gate=memory.get('gate', False))
        job.update(extra)
        identity = {k: job[k] for k in ('panel', 'seed', 'encoder', 'encoder_variant',
                                        'order', 'task', 'teaching')}
        identity['memory'] = {k: memory[k] for k in ('name', 'kind', 'params', 'feature')}
        identity['runtime'] = {k: job[k] for k in ('readout_competition', 'retrieval_timing') if k in job}
        raw = json.dumps(identity, sort_keys=True, separators=(',', ':'))
        digest = hashlib.sha256(raw.encode()).hexdigest()[:14]
        job['id'] = '%s.%s.s%d.%s.%s' % (panel, encoder, seed, name, digest)
        job['comparisons'] = []
        if parent is not None:
            job['parent_id'] = parent['id']
            job['comparisons'].append(dict(parent_id=parent['id'],
                contrast=job['contrast'], gate=job['gate'],
                factors=job['comparison_factor']))
        jobs.append(job)
        signatures[(panel, seed, encoder, name, json.dumps(variant or {}, sort_keys=True),
                    order, task, json.dumps(teaching, sort_keys=True))] = job
        return job

    frozen = {}
    for encoder in encoders:
        for seed in seeds:
            for name in specs:
                job = add('frozen', seed, encoder, name)
                frozen[(seed, encoder, name)] = job
    # Resolve after all rows exist; a parent may occur later in source order.
    for (seed, encoder, name), job in frozen.items():
        parent_name = specs[name].get('parent')
        if parent_name in specs:
            parent = frozen[(seed, encoder, parent_name)]
            job['parent_id'] = parent['id']
            job['comparisons'].append(dict(parent_id=parent['id'], contrast=specs[name]['contrast'],
                                          gate=specs[name].get('gate', False), factors=['memory']))
        additional = specs[name].get('additional_parent')
        if additional and additional != parent_name:
            job['comparisons'].append(dict(parent_id=frozen[(seed, encoder, additional)]['id'],
                contrast='protection_vs_unprotected', gate=True, factors=['memory']))
        other = {'E1': 'E0', 'E3': 'E2'}.get(encoder)
        if other in encoders:
            job['comparisons'].append(dict(parent_id=frozen[(seed, other, name)]['id'],
                contrast='clean_vs_observed_target' if encoder == 'E1' else 'local_development',
                gate=False, factors=['encoder']))

    live = {}
    if 'E3' in encoders:
        live_variants = (
            ('slow', {'online_rate': .0002}),
            ('larger_rate', {'online_rate': .001}),
            ('partition_all_plastic', {'online_rate': .0002, 'isolate_fraction': .5}),
            ('stable_plastic', {'online_rate': .0002, 'isolate_fraction': .5, 'stable_fraction': .5}),
        )
        for seed in seeds:
            for name in V24_ANCHOR_NAMES:
                for label, variant in live_variants:
                    parent = (live[(seed, name, 'partition_all_plastic')] if label == 'stable_plastic'
                              else frozen[(seed, 'E3', name)])
                    job = add('live', seed, 'E3', name, variant, parent=parent,
                              contrast='encoder_' + label, comparison_factor=['encoder_variant'],
                              variant_name=label)
                    live[(seed, name, label)] = job

        circuit_variants = (
            ('homogeneous_timescales', {'timescales': 'single'}, None),
            ('stubborn_states', {'stubborn_fraction': .5}, None),
            ('coupled_gates', {'gates': 'coupled'}, None),
            ('independent_gates', {'gates': 'independent'}, 'coupled_gates'),
            ('activity12', {'active_fraction': .125}, None),
            ('activity48', {'active_fraction': .5}, None),
            ('nonnegative_exact', {'activity_code': 'nonnegative'}, None),
            ('nonnegative_inhibition', {'activity_code': 'nonnegative', 'competition': 'inhibitory'},
             'nonnegative_exact'),
            ('pooled_recurrence', {'recurrence': 'pooled'}, None),
            ('coincidence_recurrence', {'recurrence': 'coincidence'}, None),
            ('recurrence_lesion', {'recurrence': 'off'}, None),
            ('local_recurrent_weights', {'online_rate': .0002, 'recurrent_learning': True}, 'live_slow'),
            ('local_decorrelation_fixed', {'online_rate': .0002, 'inhibitory_fixed': True}, 'live_slow'),
            ('local_decorrelation', {'online_rate': .0002, 'inhibitory_fixed': True,
                                     'inhibitory_learning': True}, 'local_decorrelation_fixed'),
            ('local_contact_retained', {'online_rate': .0002, 'turnover': 'retained'}, 'live_slow'),
            ('local_contact_turnover', {'online_rate': .0002, 'turnover': 'local'}, 'local_contact_retained'),
            ('continuous_private_state', {'reset': 'continuous'}, None),
            ('encoder_float32', {'precision': 'float32'}, None),
            ('encoder_quantized8', {'precision': 'quantized8'}, 'encoder_float32'),
        )
        for seed in seeds:
            for name in ('ordinary', 'bank8_fixed'):
                made = {}
                for label, variant, p in circuit_variants:
                    parent = (live[(seed, name, 'slow')] if p == 'live_slow' else
                              made[p] if p else frozen[(seed, 'E3', name)])
                    made[label] = add('circuit', seed, 'E3', name, variant, parent=parent,
                                      contrast=label, comparison_factor=['encoder_variant'],
                                      variant_name=label)
                add('circuit', seed, 'E3', name, parent=frozen[(seed, 'E3', name)],
                    contrast='recurrent_readout_competition', comparison_factor=['readout_competition'],
                    memory_overrides={'readout_competition': .2},
                    variant_name='recurrent_readout_competition')
                float32 = add('circuit', seed, 'E3', name,
                    parent=frozen[(seed, 'E3', name)],
                    contrast='readout_float32', comparison_factor=['memory_precision'],
                    memory_overrides={'memory_precision': 'float32'},
                    variant_name='readout_float32')
                add('circuit', seed, 'E3', name, parent=float32,
                    contrast='readout_quantized8', comparison_factor=['memory_precision'],
                    memory_overrides={'memory_precision': 'quantized8'},
                    variant_name='readout_quantized8',
                    precision_note='Plastic output values use actual int8 storage; '
                    'router and intrinsic states retain their separately counted dtype.')
            add('circuit', seed, 'E3', 'retrieve_hard32',
                parent=frozen[(seed, 'E3', 'retrieve_hard32')],
                contrast='event_triggered_retrieval', comparison_factor=['retrieval_timing'],
                memory_overrides={'retrieval_timing': 'event'},
                variant_name='event_triggered_retrieval')

        for seed in seeds:
            for name in ('ordinary', 'bank8_local_neuron_release'):
                control = add('noise', seed, 'E3', name, task='noise_control',
                              contrast='unchanged_noise_unchanged_label_control')
                for task in ('noise_only', 'noise_revision'):
                    add('noise', seed, 'E3', name, task=task, parent=control,
                        contrast='sensory_noise' if task == 'noise_only' else 'true_revision',
                        comparison_factor=['task'])
            for name in ('ordinary', 'bank8_fixed', 'bank8_local', 'allocate128_local_recency'):
                structured = add('structured', seed, 'E3', name, task='structured',
                    contrast='unseen_ordered_pair_generalization')
                add('structured', seed, 'E3', name, task='structured_order_erased',
                    parent=structured, contrast='order_information_lesion', comparison_factor=['task'])

        base_teaching = dict(trace_slots=32, gamma=.9, epsilon=.1,
                             eligibility_address='broadcast',
                             read_mode='hard', recent_bias=0.)
        teaching_conditions = [('target_immediate', dict(mode='target', delay=0,
                                                         regime='isolated', eligibility='intact'))]
        for mode in ('target', 'reward'):
            if mode == 'reward':
                teaching_conditions.append(('reward_immediate', dict(mode=mode, delay=0,
                    regime='isolated', eligibility='intact')))
            for regime in ('isolated', 'intervening'):
                for eligibility in ('intact', 'cut', 'shuffled'):
                    teaching_conditions.append(('%s_%s_%s' % (mode, regime, eligibility),
                        dict(mode=mode, delay=3, regime=regime, eligibility=eligibility)))
        for seed in seeds:
            for name in ('ordinary', 'bank8_fixed', 'allocate128_local_recency'):
                made = {}
                for label, condition in teaching_conditions:
                    expression = ([(r, b) for r in ('hard', 'soft') for b in (0., .35)]
                                  if name == 'bank8_fixed' and condition['mode'] == 'reward' else
                                  [('hard', 0.)])
                    for read_mode, bias in expression:
                        teaching = dict(base_teaching, **condition)
                        teaching.update(read_mode=read_mode, recent_bias=bias)
                        key = (label, read_mode, bias)
                        if condition['eligibility'] != 'intact':
                            p = ('%s_%s_intact' % (condition['mode'], condition['regime']), read_mode, bias)
                        elif condition['delay']:
                            p = (condition['mode'] + '_immediate', read_mode, bias)
                        elif condition['mode'] == 'reward':
                            p = ('target_immediate', 'hard', 0.)
                        else:
                            p = None
                        parent = made.get(p)
                        job = add('teaching', seed, 'E3', name, teaching=teaching, parent=parent,
                            contrast=label, comparison_factor=['teaching'], variant_name=label,
                            teaching_information_note='Scalar reward exposes only chosen action reward; '
                            'equal trials do not imply equal target information. '
                            'Allocation teaching uses sensory admission with zero values '
                            'followed by local feedback correction, including its immediate '
                            'control; this differs from main-panel one-shot allocation.')
                        if name == 'bank8_fixed' and condition['mode'] == 'reward':
                            expression_parent = made.get((label, 'hard', 0.))
                            if expression_parent is not None:
                                job['comparisons'].append(dict(
                                    parent_id=expression_parent['id'],
                                    contrast='reward_expression_trajectory', gate=False,
                                    factors=['teaching.read_mode', 'teaching.recent_bias']))
                        made[key] = job
                delayed_parent = made[('target_intervening_intact', 'hard', 0.)]
                known_delay = _v24_copy.deepcopy(delayed_parent['teaching'])
                known_delay['eligibility_address'] = 'known_delayline'
                add('teaching', seed, 'E3', name, teaching=known_delay,
                    parent=delayed_parent, contrast='known_delay_timing_prior',
                    comparison_factor=['teaching.eligibility_address'],
                    variant_name='target_intervening_known_delayline',
                    teaching_information_note='Explicit timing-prior control: fixed known '
                    'delay selects a bounded local trace by its own birth time. '
                    'The environment supplies no cue identity or stored-trace pointer.')

        if not getattr(cfg, 'no_confirmation', False):
            confirmation_seeds = [int(s) for s in _v24_selected(
                getattr(cfg, 'confirmation_seeds', None), V24_DEFAULT_CONFIRMATION_SEEDS)]
            if set(confirmation_seeds) & set(seeds):
                raise ValueError('Confirmation seeds must be disjoint from comparison seeds.')
            for seed in confirmation_seeds:
                for order in ('original', 'changed'):
                    made = {}
                    for name in V24_ANCHOR_NAMES:
                        p = specs[name].get('parent')
                        if p not in made:
                            p = specs[name].get('additional_parent', p)
                        parent = made.get(p, made.get('ordinary'))
                        made[name] = add('confirmation', seed, 'E3', name, order=order, parent=parent,
                            contrast='fresh_seed_' + specs[name]['contrast'],
                            comparison_factor=['memory'], predeclared_confirmation=True,
                            gate=(name == 'bank8_local_neuron_release'))

    by_id = {job['id']: job for job in jobs}
    if len(by_id) != len(jobs):
        raise AssertionError('Duplicate V24 job identities.')
    keep = {job['id'] for job in jobs if job['panel'] in panels and
            (requested_arms is None or job['memory']['name'] in requested_arms)}
    # A panel subset must not silently discard its declared comparator.  Cross-
    # encoder comparisons are kept only when their encoder was requested.
    pending = list(keep)
    while pending:
        job = by_id[pending.pop()]
        for comparison in job['comparisons']:
            parent_id = comparison['parent_id']
            if (requested_arms is not None and parent_id in by_id and
                    by_id[parent_id]['memory']['name'] not in requested_arms):
                comparison['parent_excluded_reason'] = 'explicit_arms_filter'
                continue
            if parent_id in by_id and parent_id not in keep:
                keep.add(parent_id)
                pending.append(parent_id)
                by_id[parent_id]['included_as_control_dependency'] = True
    selected_jobs = [job for job in jobs if job['id'] in keep]
    if requested_arms is not None:
        for job in selected_jobs:
            job['arm_filter_policy'] = ('exact_requested_memory_names; same-memory '
                'control dependencies retained; excluded-memory contrasts reported unpaired')
    return selected_jobs


def v24_coverage():
    """Every V23 row has an explicit disposition; no family-wide rejection."""
    specs = v24_memory_specs()
    by_name = {s['name']: s for s in specs}
    replacements = {}
    for s in specs:
        for ancestor in s.get('lineage', []):
            replacements.setdefault(ancestor, []).append(s['name'])
    moved = {
        'silent_fixed': ['bank8_fixed'],
        'context_fixed': ['bank4_fixed'],
        'silent_inferred': ['bank8_local', 'bank8_legacy_inferred'],
        'context_inferred': ['bank4_local', 'bank4_legacy_inferred'],
    }
    retired = {
        'historical_ridge': 'Dense historical covariance solver: evaluator feasibility reference only.',
        'projection16_strict': 'Exact global subspace algorithm retired; local geometry tested separately.',
        'projection48_strict': 'Exact global subspace algorithm retired; local geometry tested separately.',
        'projection16_leaky': 'Exact global subspace algorithm retired; local geometry tested separately.',
        'quantized_allocate128': 'Doubled dense float64 address is not compression; actual encoder quantization is separately declared.',
    }
    rows = []
    for old in arm_manifest():
        name = old['name']
        if name in retired:
            status, targets, reason = 'retired_exact_implementation', [], retired[name]
        elif name.startswith('adapter_'):
            status, targets, reason = 'rewritten_local_rule', replacements[name], (
                'Manual hidden backprop removed; local competitive retained/recycled address rules are new candidates.')
        elif name in moved:
            status, targets, reason = 'dedicated_bank_factorial', moved[name], (
                'Bank count, address adaptation, expression and bias separated; archived inferred establishment retained.')
        elif name in replacements and name not in by_name:
            status, targets, reason = 'dedicated_allocation_policy', replacements[name], (
                'Capacity fixed; eviction variants and saturation diagnostics added.')
        elif name in by_name:
            status, targets, reason = 'retained_with_locality_declaration', [name], by_name[name]['locality']
        else:
            raise AssertionError('Missing old-arm coverage: ' + name)
        rows.append(dict(source='v23', hypothesis=name, status=status,
                         experiment_rows=targets, reason=reason))
    circuits = (
        ('multiple_timescales', 'homogeneous_timescales'),
        ('stubborn_private_states', 'stubborn_states'),
        ('separate_write_forget_gates', 'coupled_gates / independent_gates'),
        ('stable_plastic_upstream_addresses', 'partition_all_plastic / stable_plastic'),
        ('sparse_activity', 'activity12 / baseline24 / activity48'),
        ('nonnegative_activity_and_explicit_inhibition', 'nonnegative_exact / nonnegative_inhibition'),
        ('delayed_recurrence_and_coincidence', 'pooled_recurrence / coincidence_recurrence / recurrence_lesion'),
        ('existing_recurrent_synapse_learning', 'local_recurrent_weights'),
        ('local_decorrelation', 'local_decorrelation_fixed / local_decorrelation'),
        ('recurrent_readout_competition', 'recurrent_readout_competition'),
        ('local_contact_turnover', 'local_contact_retained / local_contact_turnover'),
        ('continuous_private_state', 'continuous_private_state'),
        ('stored_precision', 'encoder_float32 / encoder_quantized8; readout_float32 / readout_quantized8'),
        ('retrieval_timing', 'event_triggered_retrieval'),
        ('sensory_surprise_vs_obsolescence', 'noise_control / noise_only / noise_revision'),
        ('delayed_credit', 'isolated / intervening, intact / cut / shuffled eligibility'),
        ('scalar_reward_information', 'chosen-action reward with separately trained expression trajectories'),
        ('structured_unseen_identity_generalization', 'structured / structured_order_erased'),
    )
    for hypothesis, target in circuits:
        rows.append(dict(source='older_circuit_audit', hypothesis=hypothesis,
                         status='explicit_sensitivity', experiment_rows=[target],
                         reason='Engineering representative; a negative result does not reject the entire hypothesis.'))
    return rows


def v24_manifest_selfcheck():
    from types import SimpleNamespace
    specs = v24_memory_specs()
    cfg = SimpleNamespace(seeds=[17], encoders=['E0', 'E1', 'E2', 'E3'],
                          panels=['all'], no_confirmation=False, confirmation_seeds=[101])
    jobs = v24_jobs(cfg)
    ids = {j['id'] for j in jobs}
    assert all(c['parent_id'] in ids for j in jobs for c in j['comparisons'])
    assert len([r for r in v24_coverage() if r['source'] == 'v23']) == 52
    assert len([s for s in specs if s['family'] == 'within_bank_protection']) == 24
    assert len([s for s in specs if s['family'] == 'bounded_allocation_repair']) == 10
    assert len([s for s in specs if s['family'] == 'total_memory_budget']) == 6
    assert len([j for j in jobs if j['panel'] == 'frozen']) == 404
    assert len([j for j in jobs if j['panel'] == 'confirmation']) == 18
    assert all(j['encoder'] == 'E3' for j in jobs if j['panel'] != 'frozen')
    selected_names = ['ordinary', 'v21_input', 'v22_joint',
                      'bank4_legacy_inferred', 'bank8_legacy_inferred',
                      'allocate128_original']
    subset_cfg = SimpleNamespace(seeds=[17], encoders=['E0'], panels=['frozen'],
                                 no_confirmation=True, arms=selected_names)
    subset = v24_jobs(subset_cfg)
    assert len(subset) == len(selected_names)
    assert {j['memory']['name'] for j in subset} == set(selected_names)
    assert any(c.get('parent_excluded_reason') == 'explicit_arms_filter'
               for j in subset for c in j['comparisons'])
    subset_cfg.arms = ['not_a_v24_memory']
    try:
        v24_jobs(subset_cfg)
    except ValueError as exc:
        assert 'Unknown V24 memory' in str(exc)
    else:
        raise AssertionError('Unknown --arms name was not rejected.')
    return dict(specifications=len(specs), one_seed_jobs=len(jobs),
                old_arms_covered=52, status='passed')


# === V24 teaching ===
# V24 teaching, environment perturbation and structured-transfer fragment.
import copy as _v24_copy


def v24_perturb_stream(stream, cfg, seed, task='association', order='original'):
    """Environment-only interventions. Never pass these metadata to a learner.

    Noise stress reuses the same cue cohort and event count. `train_override`
    is phase-specific so increased noise is not accidentally present during
    acquisition. Evaluation always uses the unperturbed held-out observations.
    """
    out = _v24_copy.deepcopy(stream)
    task = str(task or 'association')
    if order == 'changed':
        for p, ph in enumerate(out['schedule']):
            for c, cp in enumerate(ph['checkpoints']):
                r = rng_for(seed, 'v24_changed_order_%d_%d' % (p, c))
                ev = cp['events']
                cp['events'] = [ev[int(j)] for j in r.permutation(len(ev))]
    if task in ('noise_control', 'noise_only', 'noise_revision'):
        is_label_revision = task == 'noise_revision'
        original = np.asarray(out['original'], dtype=int)
        for p, ph in enumerate(out['schedule']):
            affected = bool(ph.get('revision', False))
            ph['stress_cohort'] = affected
            ph['sensory_noise_stress'] = bool(affected and task == 'noise_only')
            if not is_label_revision:
                ph['targets'] = original.copy()
                ph['ever_revised'] = np.zeros(len(original), dtype=bool)
                ph['revision'] = False
                for cp in ph['checkpoints']:
                    cp['events'] = [(int(i), int(v), int(original[i]))
                                    for i, v, _ in cp['events']]
            if affected and task == 'noise_only':
                x = np.array(out['train'], copy=True)
                r = rng_for(seed, 'v24_noise_stress_%d' % p)
                # Existing noise SD is cfg.noise; independent extra noise
                # gives three times that total SD on occupied pulse bins.
                extra = float(cfg.noise) * (8.0 ** .5)
                ids = np.asarray(ph['ids'], dtype=int)
                for left, right in ((0, 3), (5, 8)):
                    # One extra noisy pulse realization per sensory variant,
                    # repeated across its three occupied time bins.
                    z = r.normal(0.0, extra,
                                 (len(ids), x.shape[1], 1, x.shape[-1]))
                    x[ids, :, left:right, :] += z.astype(x.dtype)
                ph['train_override'] = x
        out['perturbation'] = dict(task=task, total_noise_multiplier=3.0
                                  if task == 'noise_only' else 1.0,
                                  affected_cohort='original revision cohorts',
                                  learner_phase_signal=False)
    elif task in ('structured_order_erased', 'structured_erased'):
        for key in ('train', 'probe'):
            x = np.array(out[key], copy=True)
            a = x[:, :, 0:3, :].copy()
            b = x[:, :, 5:8, :].copy()
            x[:, :, 0:3, :] = .5 * (a + b)
            x[:, :, 5:8, :] = .5 * (a + b)
            out[key] = x
        out['order_control'] = 'Both pulse positions receive the same mean pulse.'
    out['ordering'] = str(order)
    h = hashlib.sha256()
    for ph in out['schedule']:
        for cp in ph['checkpoints']:
            h.update(np.asarray(cp['events'], dtype='<i8').tobytes())
        if 'train_override' in ph:
            h.update(ph['train_override'].tobytes())
    h.update(out['train'].tobytes())
    out['full_stream_sha256'] = h.hexdigest()
    return out


def v24_structured_stream(cfg, seed):
    """Ordered category rule with disjoint continuous pulse-pair identities.

    Labels are 2*first_category + second_category. The category of a pulse
    is the sign of its first coordinate; all other coordinates are fresh
    nuisance variation. Neither a category ID nor a cue ID enters the model.
    Each probe has an entirely new underlying pair, rather than another noisy
    rendering of a training pair. Revision phases become refresh phases so
    this task retains one fixed generalizable rule throughout the stream.
    """
    out = build_stream(cfg, seed)
    n = len(out['original'])
    labels = np.asarray(out['original'], dtype=int)
    train_bases = cue_bases(rng_for(seed, 'v24_structured_train_pairs'), n)
    r = rng_for(seed, 'v24_structured_category_strength')
    bits = np.stack((labels // 2, labels % 2), axis=1)
    train_bases[:, :, 0] = (2 * bits - 1) * r.uniform(.75, 1.15, (n, 2))
    out['train'] = render_cues(train_bases, cfg.variants,
                              rng_for(seed, 'v24_structured_train_noise'), cfg.noise)
    # Each probe variant gets its own independent latent pair identity.
    pb = cue_bases(rng_for(seed, 'v24_structured_unseen_pairs'),
                   n * cfg.probe_variants)
    repeated_bits = np.repeat(bits, cfg.probe_variants, axis=0)
    pb[:, :, 0] = (2 * repeated_bits - 1) * r.uniform(.75, 1.15, (len(pb), 2))
    out['probe'] = render_cues(pb, 1,
                              rng_for(seed, 'v24_structured_probe_noise'),
                              cfg.noise).reshape(n, cfg.probe_variants, 12, 32)
    for ph in out['schedule']:
        if ph.get('revision', False):
            ph['name'] = 'structured_refresh_' + ph['name']
        ph['revision'] = False
        ph['ever_revised'] = np.zeros(n, dtype=bool)
        ph['targets'] = labels.copy()
        for cp in ph['checkpoints']:
            cp['events'] = [(int(i), int(v), int(labels[i]))
                            for i, v, _ in cp['events']]
    swapped = out['probe'].copy()
    swapped[:, :, 0:3, :] = out['probe'][:, :, 5:8, :]
    swapped[:, :, 5:8, :] = out['probe'][:, :, 0:3, :]
    swapped_labels = 2 * (labels % 2) + labels // 2
    out['structured_controls'] = dict(probe_swapped=swapped,
                                      targets_swapped=swapped_labels,
                                      order_sensitive_mask=(labels == 1) | (labels == 2))
    out['task_metadata'] = dict(
        task='ordered_category_rule', rule='2*first_sign_category + second_sign_category',
        training_pair_identities=n, probe_pair_identities=n * cfg.probe_variants,
        train_probe_latent_identity_overlap=0, revision_phases='same-rule refresh',
        unique_training_inputs=n * cfg.variants,
        explanation='Probe identities are new independent pulse pairs, conditional only on categories.')
    out['prefix_check'] = dict(status='not_applicable', reason='Different structured transfer task')
    return v24_perturb_stream(out, cfg, seed, 'structured', 'original')


class V24TeachingState:
    """Bounded, factorized eligibility and action tags attached to write sites.

    h is a factorized presynaptic eligibility vector, destination is a physical
    bank/slot index, and generation prevents a recycled cell receiving an old
    cell's teaching. No cue identity, label, future observation or graph is
    stored in this learner. Labels waiting for delivery reside in the explicit
    environment event queue, not in eligibility. The environment returns no
    address, ticket or cue identifier with feedback. The main intervening-cue
    condition broadcasts feedback over bounded causal eligibility. A separately
    declared known-delayline control selects by this circuit's own clock and
    fixed latency; it assumes exact timing knowledge and is not general credit.
    """
    def __init__(self, model, seed, options):
        self.model = model
        self.options = dict(options)
        self.mode = self.options.get('mode', 'target')
        self.regime = self.options.get('regime', 'isolated')
        self.trace_control = self.options.get('eligibility', 'intact')
        self.addressing = self.options.get('eligibility_address', 'broadcast')
        self.delay = max(0, int(self.options.get('delay', 0)))
        self.capacity = max(1, int(self.options.get('trace_slots', 32)))
        self.gamma = float(self.options.get('gamma', .9))
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError('Eligibility gamma must lie in [0,1].')
        self.epsilon = float(self.options.get('epsilon', .1))
        if self.mode not in ('target', 'reward'):
            raise ValueError('Teaching mode must be target or reward.')
        if self.regime not in ('isolated', 'intervening'):
            raise ValueError('Teaching regime must be isolated or intervening.')
        if self.trace_control not in ('intact', 'cut', 'shuffled'):
            raise ValueError('Eligibility control must be intact, cut or shuffled.')
        if self.addressing not in ('broadcast', 'known_delayline'):
            raise ValueError('Eligibility addressing must be broadcast or known_delayline.')
        if not 0.0 <= self.epsilon <= 1.0:
            raise ValueError('Action exploration probability must lie in [0,1].')
        self.rng = np.random.default_rng(int(seed))
        d = int(model.d)
        self.h = np.zeros((self.capacity, d), dtype=np.float64)
        self.pred = np.zeros((self.capacity, 4), dtype=np.float64)
        self.dest = np.full(self.capacity, -1, dtype=np.int64)
        self.generation = np.full(self.capacity, -1, dtype=np.int64)
        self.action = np.zeros(self.capacity, dtype=np.int64)
        self.born = np.full(self.capacity, -1, dtype=np.int64)
        self.present = np.zeros(self.capacity, dtype=bool)
        self.next_slot = 0
        self.events = self.feedback_events = self.trace_lost = 0
        self.stale_destination = self.shuffle_unavailable = self.shuffled_events = 0
        self.cuts = self.exploratory = 0
        self.reward_sum = self.update_norm_sum = self.retained_trace_sum = 0.0
        self.eligible_count_sum = self.feedback_gain_sum = 0.0
        self.clock = 0
        self.is_allocation = hasattr(model, 'values') and hasattr(model, 'centers')
        self.is_bank = hasattr(model, 'route_probabilities') and not self.is_allocation
        if self.is_bank and (getattr(model, 'feature_index', None) is not None or
                             getattr(model, 'importance', None) is not None or
                             float(getattr(model, 'spillover', 0.0)) != 0.0):
            raise ValueError('Teaching bank anchors must have full-width unprotected isolated writes.')
        self.last_sensory_info = {}

    def arrays(self):
        return {k: v for k, v in vars(self).items() if isinstance(v, np.ndarray)}

    def output(self, H):
        if self.is_bank:
            return self.model.output_view(
                H, read_mode=self.options.get('read_mode', 'hard'),
                recent_bias=float(self.options.get('recent_bias', 0.0)))
        return self.model.output(H)

    def begin(self, h, clock):
        h = np.asarray(h, dtype=np.float64).reshape(-1)
        self.clock = int(clock)
        scores = np.asarray(self.output(h), dtype=np.float64).reshape(4)
        exploratory = bool(self.mode == 'reward' and self.rng.random() < self.epsilon)
        action = int(self.rng.integers(4)) if exploratory else int(np.argmax(scores))
        self.exploratory += int(exploratory)
        destination = 0
        generation = -1
        self.last_sensory_info = {}
        if self.is_allocation:
            if not hasattr(self.model, 'begin_teaching'):
                raise TypeError('Delayed allocation requires begin_teaching(h).')
            info = self.model.begin_teaching(h)
            self.last_sensory_info = dict(info)
            destination = int(info.get('selected_slot', -1))
            if destination >= 0:
                generation = int(info.get('generation', self.model.born[destination]))
                pred = np.array(self.model.values[destination], copy=True)
            else:
                pred = np.zeros(4)
        elif self.is_bank:
            p = np.asarray(self.model.route_probabilities(h, read_mode='hard'))
            destination = int(np.argmax(p))
            pred = np.asarray(h @ self.model.V[destination], dtype=np.float64)
            if hasattr(self.model, 'counts'):
                self.model.counts[destination] += 1
            if getattr(self.model, 'routing', 'fixed') == 'local':
                rate = min(float(self.model.address_eta),
                           1.0 / float(self.model.counts[destination]))
                self.model.centers[destination] = unit_rows(
                    (1.0 - rate) * self.model.centers[destination] + rate * unit_rows(h))
            if hasattr(self.model, 'route_changes'):
                previous = int(getattr(self.model, 'active', -1))
                self.model.route_changes += int(previous >= 0 and previous != destination)
            self.model.t += 1
            if hasattr(self.model, 'active'):
                self.model.active = destination
            if hasattr(self.model, 'last_seen'):
                self.model.last_seen[destination] = self.model.t
            if hasattr(self.model, 'last_bank'):
                self.model.last_bank = destination
            if hasattr(self.model, 'last_active'):
                self.model.last_active = destination
        else:
            pred = scores.copy()
            self.model.t += 1
        slot = self.next_slot
        if self.present[slot]:
            self.trace_lost += 1
        self.h[slot] = h
        self.pred[slot] = pred
        self.dest[slot] = destination
        self.generation[slot] = generation
        self.action[slot] = action
        self.born[slot] = int(clock)
        self.present[slot] = True
        self.next_slot = (slot + 1) % self.capacity
        self.events += 1
        return action

    def deliver(self, outcome, clock):
        """Receive only a teaching outcome and intrinsic simulation time.

        There is no environment-supplied return address. For delayed intervening
        cues, every retained causal trace is eligible in the main condition.
        A fixed feedback-gain circuit caps total eligibility mass at one; it
        never uses old targets or probe damage. This leaves isolated-trial
        decay uncompensated while avoiding a larger broadcast write budget.
        """
        self.clock = int(clock)
        self.feedback_events += 1
        if self.mode == 'reward':
            reward = float(outcome)
            if reward not in (0.0, 1.0):
                raise ValueError('Chosen-action reward must be scalar zero or one.')
            self.reward_sum += reward
        available = np.flatnonzero(self.present & (self.born <= int(clock)))
        stale_at_entry = 0
        if self.is_allocation and len(available):
            physical = available[self.dest[available] >= 0]
            generations = getattr(self.model, 'generation', self.model.born)
            stale = physical[generations[self.dest[physical]] != self.generation[physical]]
            stale_at_entry = len(stale)
            self.stale_destination += stale_at_entry
            self.present[stale] = False
            # Dead or rejected write sites contribute no inhibition/feedback
            # demand; exclude them before the common gain normalization.
            available = physical[self.present[physical]]
        if self.addressing == 'known_delayline':
            selected = available[self.born[available] == int(clock) - self.delay]
        elif self.regime == 'isolated' or self.delay == 0:
            # Immediate teaching coincides with the currently active sensory
            # trace. Isolated trials explicitly provide a sensory-trial boundary.
            selected = (np.asarray([available[np.argmax(self.born[available])]], dtype=int)
                        if len(available) else available)
        else:
            selected = available.copy()
        if not len(selected):
            return dict(update_norm=0.0, trace_available=False,
                        stale_destination=bool(stale_at_entry))
        retention = np.power(self.gamma, np.maximum(0, int(clock) - self.born[selected]))
        if self.trace_control == 'shuffled':
            if len(selected) > 1:
                # Keep the complete decay-weight multiset and shuffle its
                # temporal assignment to intact physical trace contents.
                retention = retention[self.rng.permutation(len(retention))]
            else:
                earlier = available[available != selected[0]]
                if not len(earlier):
                    self.shuffle_unavailable += 1
                    return dict(update_norm=0.0, trace_available=False)
                # Substitute a past trace's content with the SAME current
                # eligibility age/weight, avoiding an attenuation confound.
                selected = np.asarray([self.rng.choice(earlier)], dtype=int)
            self.shuffled_events += 1
        mass = float(np.sum(retention))
        feedback_gain = 1.0 / max(1.0, mass)
        self.retained_trace_sum += mass
        self.eligible_count_sum += len(selected)
        self.feedback_gain_sum += feedback_gain
        if self.trace_control == 'cut':
            self.cuts += 1
            return dict(update_norm=0.0, trace_available=True, trace_retention=0.0)
        eta = float(getattr(self.model, 'eta', .6))
        aggregate = np.zeros_like(self.model.values if self.is_allocation else self.model.V)
        stale = stale_at_entry
        for slot, weight in zip(selected, retention):
            destination = int(self.dest[slot])
            if destination < 0:
                continue
            if self.is_allocation:
                generations = getattr(self.model, 'generation', self.model.born)
                if int(generations[destination]) != int(self.generation[slot]):
                    self.stale_destination += 1
                    stale += 1
                    continue
            if self.mode == 'reward':
                # Reward is broadcast; only locally retained actual-action tags
                # can receive it. No correct four-label vector is constructed.
                error = np.zeros(4, dtype=np.float64)
                a = int(self.action[slot])
                error[a] = reward - float(self.pred[slot, a])
            else:
                error = target4(int(outcome)) - self.pred[slot]
            scale = eta * float(weight) * feedback_gain
            if self.is_allocation:
                aggregate[destination] += scale * error
                continue
            h = self.h[slot]
            change = scale * np.outer(h, error) / max(float(h @ h), 1e-12)
            if self.is_bank:
                aggregate[destination] += change
            else:
                aggregate += change
        if self.is_allocation:
            self.model.values += aggregate
        else:
            self.model.V += aggregate
        norm = float(np.linalg.norm(aggregate))
        self.update_norm_sum += norm
        return dict(update_norm=norm, trace_available=True, trace_retention=mass,
                    eligible_trace_count=len(selected), feedback_gain=feedback_gain,
                    stale_destination=bool(stale))

    def diagnostics(self):
        return dict(external_events=self.events, feedback_events=self.feedback_events,
                    simulation_ticks=self.clock, eligibility_slots=self.capacity,
                    overwritten_eligibility_traces=self.trace_lost,
                    stale_recycled_destination=self.stale_destination,
                    shuffled_teaching_events=self.shuffled_events,
                    shuffle_unavailable=self.shuffle_unavailable,
                    cut_teaching_events=self.cuts,
                    exploratory_actions=self.exploratory,
                    mean_observed_reward=self.reward_sum / max(1, self.feedback_events)
                    if self.mode == 'reward' else None,
                    mean_update_norm=self.update_norm_sum / max(1, self.feedback_events),
                    mean_retained_eligibility=self.retained_trace_sum / max(1, self.feedback_events),
                    mean_eligible_trace_count=self.eligible_count_sum / max(1, self.feedback_events),
                    mean_feedback_gain=self.feedback_gain_sum / max(1, self.feedback_events),
                    eligibility_bytes=sum(a.nbytes for a in self.arrays().values()),
                    eligibility_address=self.addressing,
                    temporal_credit='Intrinsic fixed-delay timing control' if self.addressing == 'known_delayline'
                    else 'Causal bounded broadcast eligibility; environment supplies no return address')


def v24_run_teaching_job(cfg, job, encoder, stream, make_model):
    """A causal, bounded teaching protocol, with phase-level evaluator snapshots.

    Intervening trials deliver due outcomes before the next cue and are flushed
    at checkpoint boundaries, so checkpoint scores always include the same
    number of received outcomes. Flush blank ticks are counted explicitly.
    Environment queue holds only (due clock, outcome); no hidden
    label vector enters a reward learner. The finite queue size is reported.
    """
    options = dict(job.get('teaching') or {})
    model = make_model()
    state = V24TeachingState(model, int(job['seed']) + 90217, options)
    result = dict(metrics=[], diagnostics=[], allocation_events=[], costs={}, failures=[])
    queue = []
    clock = external = peak_queue = blank_ticks = 0
    feature = job.get('memory', {}).get('feature', 'sparse')
    probe = stream['probe']
    H = encoder.features(probe.reshape(-1, 12, 32), feature=feature, adapt=False)
    H = np.asarray(H).reshape(len(probe), probe.shape[1], -1)
    encoder_forwards = len(probe) * probe.shape[1]

    def receive_due(now):
        ready = [q for q in queue if q[0] <= now]
        queue[:] = [q for q in queue if q[0] > now]
        for _, payload in ready:
            state.deliver(payload, now)

    for ph in stream['schedule']:
        inputs = ph.get('train_override', stream['train'])
        for cp in ph['checkpoints']:
            for i, variant, environment_label in cp['events']:
                if state.regime == 'intervening':
                    receive_due(clock)
                h = encoder.features(inputs[int(i), int(variant)], feature=feature, adapt=False)
                h = np.asarray(h).reshape(-1)
                encoder_forwards += 1
                action = state.begin(h, clock)
                sensory = state.last_sensory_info
                if state.is_allocation and any(sensory.get(k, 0)
                                              for k in ('allocated', 'recycled', 'rejected')):
                    # Identity appears only in this evaluator record, after the
                    # sensory-only admission decision has already happened.
                    event_row = dict(sensory)
                    event_row.update(phase=ph['name'], pass_index=int(cp['pass_index']),
                                     external=external + 1, identity=int(i),
                                     simulation_tick=clock)
                    result['allocation_events'].append(event_row)
                # This comparison is performed by the environment, after action.
                payload = (float(action == int(environment_label))
                           if state.mode == 'reward' else int(environment_label))
                queue.append((clock + state.delay, payload))
                peak_queue = max(peak_queue, len(queue))
                external += 1
                if state.regime == 'isolated':
                    for tick in range(state.delay + 1):
                        receive_due(clock + tick)
                    clock += state.delay + 1
                    blank_ticks += state.delay
                else:
                    receive_due(clock)
                    clock += 1
            # Standardized post-feedback evaluation. No learner phase signal.
            while queue:
                receive_due(clock)
                clock += 1
                blank_ticks += 1
            before = model.state_digest()
            z = np.asarray(state.output(H.reshape(-1, H.shape[-1])))
            if before != model.state_digest():
                raise AssertionError('Teaching evaluation mutated memory.')
            if not np.isfinite(z).all():
                raise FloatingPointError('Nonfinite teaching scores.')
            n = len(H)
            seen = np.asarray(ph['seen'], dtype=bool)
            active = np.zeros(n, dtype=bool)
            active[np.asarray(ph['ids'], dtype=int)] = True
            idx = np.arange(n)
            a = idx < cfg.group_cues
            b = (idx >= cfg.group_cues) & (idx < 2 * cfg.group_cues)
            c = (idx >= 2 * cfg.group_cues) & (idx < 3 * cfg.group_cues)
            masks = dict(all_current=seen, phase_current=active & seen,
                         prior_current=seen & ~active,
                         A_current=a & seen, B_current=b & seen, C_current=c & seen,
                         initial_revised=np.asarray(stream['initial_reverse']) & seen,
                         original_never_revised=seen & ~np.asarray(ph['ever_revised']),
                         initial_unchanged=(a | b) & ~np.asarray(stream['initial_reverse']) & seen)
            row = dict(phase=ph['name'], revision=bool(ph.get('revision', False)),
                       pass_index=int(cp['pass_index']), external=external,
                       simulation_ticks=clock, view='teaching_action_view',
                       teaching_mode=state.mode, teaching_regime=state.regime,
                       teaching_delay=state.delay, eligibility=state.trace_control,
                       eligibility_address=state.addressing,
                       seen_identities=int(seen.sum()))
            row.update({k: accuracy(z, ph['targets'], mask, probe.shape[1])
                        for k, mask in masks.items()})
            row['obsolete_association_accuracy'] = accuracy(
                z, stream['original'], seen & np.asarray(ph['ever_revised']), probe.shape[1])
            result['metrics'].append(row)
            diagnostics = state.diagnostics()
            diagnostics.update(phase=ph['name'], pass_index=int(cp['pass_index']),
                               blank_simulation_ticks=blank_ticks,
                               pending_environment_queue_peak=peak_queue)
            result['diagnostics'].append(diagnostics)
    costs = dict(external_presentations=external,
                 extra_labeled_presentations=0, extra_presentations=0,
                 teacher_messages=state.feedback_events,
                 teacher_channel_count=1 if state.mode == 'reward' else 4,
                 teacher_information='scalar chosen-action reward' if state.mode == 'reward'
                 else 'full four-output target delivered after stated delay',
                 simulation_ticks=clock, blank_simulation_ticks=blank_ticks,
                 pending_environment_queue_peak=peak_queue,
                 pending_environment_payload_bytes_peak=peak_queue * 16,
                 environment_queue_excluded_from_learner_memory=True,
                 readout_mutable_bytes=model.mutable_bytes(),
                 eligibility_mutable_bytes=sum(a.nbytes for a in state.arrays().values()),
                 learner_mutable_bytes=model.mutable_bytes() +
                 sum(a.nbytes for a in state.arrays().values()),
                 encoder_forwards=encoder_forwards,
                 eligibility_address=state.addressing,
                 feedback_gain_rule='1/max(1,total eligible decay mass); no old-target/probe access',
                 environment_feedback_fields='due simulation time and teaching outcome only',
                 protocol_note='Frozen encoder; local readout teaching. Main delayed intervening condition broadcasts over bounded causal traces; intrinsic known_delayline is a separate exact-latency timing-prior control. Immediate feedback uses current sensory activity; isolated trials provide a trial boundary. Allocation admits sensory patterns with zero values before feedback, then applies the same eta-scaled correction for new and matched slots, unlike main supervised one-shot new-slot assignment. Stale physical-slot traces are invalidated. No environment return address or cue identity is supplied.')
    if hasattr(model, 'fixed_arrays'):
        costs['readout_fixed_bytes'] = sum(a.nbytes for a in model.fixed_arrays().values())
    result['costs'] = costs
    return result


def v24_teaching_checks():
    report = {}
    h = np.array([1.0, 0.0, 0.0, 0.0])
    plain = PlainMemory(4, 1)
    state = V24TeachingState(plain, 2, dict(mode='target', delay=0))
    action = state.begin(h, 0)
    state.deliver(2, 0)
    ref = PlainMemory(4, 1)
    ref.observe(h, 2)
    assert np.allclose(plain.V, ref.V)
    report['immediate_target_matches_plain_delta'] = True
    reward_model = PlainMemory(4, 1)
    reward = V24TeachingState(reward_model, 2, dict(mode='reward', epsilon=0.0))
    action = reward.begin(h, 0)
    reward.deliver(1.0, 0)
    assert np.count_nonzero(np.any(reward_model.V != 0, axis=0)) == 1
    report['reward_writes_only_chosen_action'] = True
    delayed_model = PlainMemory(4, 1)
    delayed = V24TeachingState(delayed_model, 2, dict(mode='target', gamma=.5))
    delayed.begin(h, 0)
    delayed.deliver(2, 3)
    assert np.allclose(delayed_model.V, ref.V * .125)
    report['delay_decays_eligibility_without_compensation'] = True
    cut_model = PlainMemory(4, 1)
    cut = V24TeachingState(cut_model, 2, dict(mode='target', eligibility='cut'))
    cut.begin(h, 0)
    cut.deliver(2, 0)
    assert not np.any(cut_model.V)
    report['cut_prevents_readout_update'] = True
    bounded = V24TeachingState(PlainMemory(4, 1), 2,
                              dict(trace_slots=2, delay=3,
                                   eligibility_address='known_delayline'))
    bounded.begin(h, 0)
    bounded.begin(h, 1)
    bounded.begin(h, 2)
    assert not bounded.deliver(1, 3)['trace_available']
    assert bounded.h.shape == (2, 4) and bounded.trace_lost == 1
    report['bounded_delayline_overwrite_loses_intrinsic_timed_trace'] = True
    broadcast_model = PlainMemory(4, 1)
    broadcast = V24TeachingState(broadcast_model, 2,
                                 dict(delay=3, regime='intervening'))
    broadcast.begin(h, 0)
    broadcast.begin(np.array([0., 1., 0., 0.]), 1)
    info = broadcast.deliver(2, 3)
    assert info['eligible_trace_count'] == 2
    assert np.linalg.norm(broadcast_model.V[0]) > 0
    assert np.linalg.norm(broadcast_model.V[1]) > 0
    report['delayed_broadcast_has_no_environment_return_address'] = True
    timed_model = PlainMemory(4, 1)
    timed = V24TeachingState(timed_model, 2,
                             dict(delay=3, regime='intervening',
                                  eligibility_address='known_delayline'))
    timed.begin(h, 0)
    timed.begin(np.array([0., 1., 0., 0.]), 1)
    timed.deliver(2, 3)
    assert np.linalg.norm(timed_model.V[0]) > 0 and not np.any(timed_model.V[1])
    report['known_delayline_selects_by_intrinsic_time_only'] = True
    other = np.array([0., 1., 0., 0.])
    controls = [V24TeachingState(PlainMemory(4, 1), 2,
                                 dict(delay=3, regime='intervening', eligibility=c))
                for c in ('intact', 'shuffled')]
    packets = []
    for control in controls:
        control.begin(h, 0)
        control.begin(other, 1)
        packets.append(control.deliver(2, 3))
    assert np.isclose(packets[0]['trace_retention'], packets[1]['trace_retention'])
    assert np.isclose(packets[0]['feedback_gain'], packets[1]['feedback_gain'])
    assert np.isclose(packets[0]['update_norm'], packets[1]['update_norm'])
    report['temporal_shuffle_preserves_decay_mass_and_orthogonal_write_norm'] = True
    if 'V24AllocationMemory' in globals():
        allocator = V24AllocationMemory(4, 1, slots=1, policy='local_recency')
        alloc = V24TeachingState(allocator, 2, dict(delay=3, regime='intervening'))
        alloc.begin(h, 0)
        alloc.begin(other, 1)
        packet = alloc.deliver(2, 3)
        assert packet['stale_destination'] and packet['eligible_trace_count'] == 1
        assert packet['feedback_gain'] == 1.0
        assert np.isclose(allocator.values[0, 2], allocator.eta * .9 ** 2)
        report['recycled_traces_removed_before_broadcast_gain'] = True
    return report


# === V24 reporting ===
# ---- V24 command line and compact reporting ---------------------------------
import gzip
import statistics


def v24_parse_config(argv=None):
    """Full specification by default; every execution filter is recorded."""
    ap = argparse.ArgumentParser(description=(
        'ELM V24: local development, banked memory and bounded allocation. '
        'One default command runs all predeclared panels and produces one results ZIP.'),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--install-deps', action='store_true', help='Install NumPy/PyTorch only if missing.')
    ap.add_argument('--out-dir', type=Path)
    ap.add_argument('--resume', type=Path, help='Resume completed individual jobs; incomplete jobs restart.')
    ap.add_argument('--seeds', type=int, nargs='+', default=[17,29,43,59,71,89])
    ap.add_argument('--confirmation-seeds', type=int, nargs='*', default=[101,127,149,173,197,223])
    ap.add_argument('--no-confirmation', action='store_true', help='Omit the separately reported fresh-seed confirmation panel.')
    ap.add_argument('--encoders', nargs='+', choices=['E0','E1','E2','E3'], default=['E0','E1','E2','E3'])
    ap.add_argument('--panels', nargs='+', choices=['all','frozen','live','circuit','teaching','structured','noise','confirmation'], default=['all'])
    ap.add_argument('--arms', nargs='+', help='Optional exact memory names; every other manifest row is marked unrequested.')
    ap.add_argument('--max-jobs', type=int, help='Execute at most this many selected jobs; mark the archive partial.')
    ap.add_argument('--list-jobs', action='store_true', help='Print the complete selected job manifest without training.')
    ap.add_argument('--list-arms', action='store_true', help='Print unique selected memory specifications without training.')
    ap.add_argument('--self-test', action='store_true', help='Arithmetic, locality and reporting checks only; no full experiment.')
    ap.add_argument('--smoke', action='store_true', help='All selected variants on a tiny execution check; not scientific results.')
    ap.add_argument('--stress-waves', type=int, default=8)
    ap.add_argument('--dev-cues', type=int, default=64)
    ap.add_argument('--dev-passes', type=int, default=6)
    ap.add_argument('--group-cues', type=int, default=24)
    ap.add_argument('--variants', type=int, default=2)
    ap.add_argument('--probe-variants', type=int, default=3)
    ap.add_argument('--passes', type=int, default=3)
    ap.add_argument('--reverse-cues', type=int, default=12)
    ap.add_argument('--noise', type=float, default=.06)
    ap.add_argument('--active-fraction', type=float, default=.25)
    ap.add_argument('--learning-rate', type=float, default=.6)
    ap.add_argument('--write-energy-floor', type=float, default=.5)
    ap.add_argument('--ridge', type=float, default=.5)
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--diagnostic-interval', type=int, default=32, help='Sample evaluator-only write effects every N events; 0 disables.')
    cfg = ap.parse_args(argv)
    cfg.neurons = 96
    if cfg.smoke:
        cfg.seeds=[17]; cfg.confirmation_seeds=[]; cfg.no_confirmation=True
        cfg.dev_cues=16; cfg.dev_passes=1; cfg.group_cues=8; cfg.passes=1
        cfg.reverse_cues=4; cfg.stress_waves=1; cfg.diagnostic_interval=8
    if cfg.no_confirmation:
        cfg.confirmation_seeds=[]
    for name in ['seeds','confirmation_seeds','encoders','panels']:
        vals=getattr(cfg,name)
        if len(set(vals)) != len(vals): ap.error('--%s contains duplicates.' % name.replace('_','-'))
    if set(cfg.seeds)&set(cfg.confirmation_seeds):
        ap.error('Comparison and confirmation seeds must be disjoint.')
    if 'all' in cfg.panels and len(cfg.panels)>1:
        ap.error('Use --panels all alone, or name individual panels.')
    if cfg.group_cues<4 or cfg.group_cues%4:
        ap.error('--group-cues must be a positive multiple of 4, at least 4.')
    if cfg.reverse_cues<2 or cfg.reverse_cues%2 or cfg.reverse_cues>2*cfg.group_cues:
        ap.error('--reverse-cues must be even and between 2 and twice group-cues.')
    if min(cfg.dev_cues,cfg.dev_passes,cfg.variants,cfg.probe_variants,cfg.passes,cfg.threads)<1:
        ap.error('Counts and threads must be positive.')
    if cfg.stress_waves<0 or cfg.diagnostic_interval<0:
        ap.error('Wave count and diagnostic interval cannot be negative.')
    if cfg.max_jobs is not None and cfg.max_jobs<1:
        ap.error('--max-jobs must be positive.')
    if not (0<cfg.active_fraction<=1 and 0<cfg.learning_rate<=1 and 0<cfg.write_energy_floor<=1):
        ap.error('Fractions and rates must be in (0, 1].')
    if cfg.ridge<=0 or cfg.noise<0 or not math.isfinite(cfg.noise):
        ap.error('Ridge must be positive; noise must be finite and nonnegative.')
    if cfg.resume and cfg.out_dir: ap.error('Use either --resume or --out-dir.')
    cfg.execution_filters = dict(encoders=cfg.encoders,panels=cfg.panels,arms=cfg.arms,
        seeds=cfg.seeds,confirmation_seeds=cfg.confirmation_seeds,max_jobs=cfg.max_jobs,
        no_confirmation=cfg.no_confirmation,smoke=cfg.smoke)
    return cfg


def v24_report_plain(value):
    """JSON-safe reporting values; nonfinite results are explicit nulls."""
    if isinstance(value,dict): return {str(k):v24_report_plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [v24_report_plain(v) for v in value]
    if isinstance(value,Path): return str(value)
    if isinstance(value,float): return value if math.isfinite(value) else None
    if value is None or isinstance(value,(str,bool,int)): return value
    if hasattr(value,'item'):
        try: return v24_report_plain(value.item())
        except (ValueError,TypeError): pass
    if hasattr(value,'tolist'): return v24_report_plain(value.tolist())
    return str(value)


def v24_report_json(path,value):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(v24_report_plain(value),indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')
    tmp.replace(path)


def v24_report_table(out,filename,rows,compress_after=5000):
    rows=[v24_report_plain(r) for r in rows]
    keys=list(dict.fromkeys(k for r in rows for k in r))
    path=Path(out)/filename
    compressed=len(rows)>compress_after
    destination=Path(str(path)+'.gz') if compressed else path
    stale=path if compressed else Path(str(path)+'.gz')
    if stale.exists(): stale.unlink()
    opener=gzip.open if compressed else open
    with opener(destination,'wt',newline='',encoding='utf-8') as f:
        if keys:
            writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
            for row in rows:
                writer.writerow({k:(json.dumps(v,sort_keys=True,separators=(',',':'))
                    if isinstance(v,(dict,list)) else v) for k,v in row.items()})
    return dict(file=destination.name,rows=len(rows),columns=keys,compressed=compressed)


def v24_report_number(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(float(value))


def v24_report_mean(values):
    values=[float(x) for x in values if v24_report_number(x)]
    return statistics.fmean(values) if values else None


def v24_report_interval(values):
    """Two-sided Student interval across independent seed differences, no SciPy."""
    values=[float(x) for x in values if v24_report_number(x)]
    n=len(values);mean=statistics.fmean(values) if n else None
    if n<2: return dict(n=n,mean=mean,sd=None,ci95_low=None,ci95_high=None)
    sd=statistics.stdev(values);df=n-1
    table={1:12.706204736,2:4.302652730,3:3.182446305,4:2.776445105,
        5:2.570581836,6:2.446911851,7:2.364624252,8:2.306004135,
        9:2.262157163,10:2.228138852,11:2.200985160,12:2.178812830,
        15:2.131449546,20:2.085963447,30:2.042272456,60:2.000297822}
    # Choose the lower available df, giving a conservative critical value.
    critical=table[max(k for k in table if k<=df)]
    half=critical*sd/math.sqrt(n)
    return dict(n=n,mean=mean,sd=sd,ci95_low=mean-half,ci95_high=mean+half)


def v24_report_job_meta(job):
    mem=job.get('memory') or {}
    return dict(job_id=job.get('id'),panel=job.get('panel'),seed=job.get('seed'),
        encoder=job.get('encoder'),arm=mem.get('name',job.get('arm')),
        family=mem.get('family'),memory_locality=mem.get('locality'),
        locality=('historical_encoder_reference' if job.get('encoder') in ['E0','E1'] else 'no_hidden_backprop_candidate_with_declared_primitives'),
        encoder_variant=json.dumps(v24_report_plain(job.get('encoder_variant') or {}),sort_keys=True,separators=(',',':')),
        task=job.get('task','association'),order=job.get('order','original'),
        teaching=json.dumps(v24_report_plain(job.get('teaching')),sort_keys=True,separators=(',',':')))


def v24_report_summaries(job,result):
    meta=v24_report_job_meta(job)
    by_view={}
    for i,row in enumerate(result.get('metrics') or []):
        if not isinstance(row,dict): continue
        view=row.get('view','default')
        if isinstance(view,(dict,list)):view=json.dumps(view,sort_keys=True)
        by_view.setdefault(str(view),[]).append(dict(row,_row_index=i))
    cost=result.get('costs') or {}
    if isinstance(cost,list):cost=cost[-1] if cost else {}
    cost=dict(cost)
    if 'total_memory_numeric_bytes' in cost:cost.setdefault('total_nonencoder_numeric_bytes',cost['total_memory_numeric_bytes'])
    if 'total_learning_presentations' not in cost:
        external=cost.get('external_presentations',cost.get('external_presentations_completed'))
        if v24_report_number(external):cost['total_learning_presentations']=external+cost.get('extra_labeled_presentations',0)
    if not by_view:by_view={'default':[]}
    summaries=[]
    for view,rows in by_view.items():
        phase_rows={}
        for row in rows:
            phase=str(row.get('phase',row.get('checkpoint','unknown')))
            # Long-form cohort rows are combined without replacing other cohorts.
            if 'cohort' in row and ('accuracy' in row or 'current_accuracy' in row):
                checkpoint=(row.get('external_presentations',0),row.get('pass_index',0))
                old=phase_rows.get(phase,{})
                if checkpoint>(old.get('external_presentations',0),old.get('pass_index',0)):
                    old={}
                old.update({k:v for k,v in row.items() if k not in ['cohort','accuracy','current_accuracy']})
                old[str(row['cohort'])]=row.get('accuracy',row.get('current_accuracy'))
                phase_rows[phase]=old
            else:
                old=phase_rows.get(phase)
                key=(row.get('external_presentations',0),row.get('pass_index',0),row['_row_index'])
                oldkey=(old.get('external_presentations',0),old.get('pass_index',0),old.get('_row_index',-1)) if old else (-1,-1,-1)
                if key>=oldkey:phase_rows[phase]=row
        phases=sorted(phase_rows.items(),key=lambda p:(p[1].get('external_presentations',0),p[1].get('_row_index',0)))
        last=phases[-1][1] if phases else {}
        novel=[r for name,r in phases if name.startswith('novel_') or name in ['acquire_A','interfere_B','interfere_C']]
        def is_revision(name,row):
            if 'revision' in row:return row['revision'] is True
            return name.startswith('revision_') or name in ['revise_subset','revise_again']
        revised=[r for name,r in phases if is_revision(name,r)]
        late_novel=novel[-3:];late_revision=revised[-3:]
        def metric(phase,key):return phase_rows.get(phase,{}).get(key)
        def revision_metric(phase,key):
            row=phase_rows.get(phase,{})
            return row.get(key) if is_revision(phase,row) else None
        failures=result.get('failures') or []
        state='failed' if failures or str(result.get('status','')).lower() in ['failed','error','incomplete'] else ('completed' if rows else 'no_metrics')
        summary=dict(meta,view=view,status=state,metric_rows=len(rows),phases=len(phases),
            terminal_phase=phases[-1][0] if phases else None,
            A_acquired=metric('acquire_A','A_current'),A_after_B=metric('interfere_B','A_current'),
            B_acquired=metric('interfere_B','B_current'),
            first_revision=revision_metric('revise_subset','phase_current'),
            first_revision_after_C=(metric('interfere_C','initial_revised') if is_revision('revise_subset',phase_rows.get('revise_subset',{})) else None),
            second_revision=revision_metric('revise_again','phase_current'),
            terminal_all_current=last.get('all_current'),terminal_prior_current=last.get('prior_current'),
            terminal_phase_current=last.get('phase_current'),
            terminal_unchanged=last.get('original_never_revised'),
            late_acquisition=v24_report_mean([r.get('phase_current',r.get('accuracy')) for r in late_novel]),
            late_revision=v24_report_mean([r.get('phase_current',r.get('accuracy')) for r in late_revision]),
            late_retention=v24_report_mean([r.get('prior_current') for r in late_novel]),
            late_novel_phases=len(late_novel),late_revision_phases=len(late_revision),
            late_novel_phase_names=[name for name,r in phases if r in late_novel],
            late_revision_phase_names=[name for name,r in phases if r in late_revision])
        for key in ['external_presentations_completed','external_presentations','extra_labeled_presentations',
            'total_labeled_presentations','peak_mutable_numeric_bytes','mutable_numeric_bytes','fixed_numeric_bytes',
            'reported_fixed_numeric_bytes','total_nonencoder_numeric_bytes','total_memory_numeric_bytes',
            'peak_memory_numeric_bytes','shared_teaching_numeric_bytes','total_learning_presentations','online_loop_seconds']:
            if key in cost:summary[key]=cost[key]
        for key in ['macro_cohort_current','all_current_macro','cohort_macro_accuracy','current_mse']:
            if key in last:summary['terminal_'+key]=last[key]
        summaries.append(summary)
    return summaries


_V24_REPORT_ENDPOINTS=['A_after_B','B_acquired','first_revision_after_C','second_revision',
    'terminal_all_current','terminal_prior_current','terminal_unchanged',
    'late_acquisition','late_revision','late_retention']


def v24_report_comparisons(jobs,summaries):
    job_by_id={j['id']:j for j in jobs};summary_by={(s['job_id'],s['view']):s for s in summaries}
    pairs=[];missing=[]
    for row in summaries:
        job=job_by_id.get(row['job_id'],{})
        declarations=list(job.get('comparisons') or [])
        if job.get('parent_id') and not any(d.get('parent_id')==job['parent_id'] for d in declarations):
            declarations.append(dict(parent_id=job['parent_id'],contrast=job.get('contrast') or job.get('memory',{}).get('contrast','declared_parent'),gate=job.get('gate',job.get('memory',{}).get('gate',False))))
        if not declarations:
            pname=job.get('memory',{}).get('parent') or job.get('parent_memory')
            if pname:
                candidates=[j for j in jobs if (j.get('memory') or {}).get('name')==pname and
                    all(j.get(k)==job.get(k) for k in ['seed','panel','encoder','task','order','teaching','encoder_variant'])]
                if len(candidates)==1:declarations=[dict(parent_id=candidates[0]['id'],contrast=job.get('memory',{}).get('contrast','memory_parent'))]
        # Pure expression views are evaluated from the identical stored state,
        # rather than charging four separate supervised learning trajectories.
        expression_parents={
            'soft_b0':[('hard_b0','hard_vs_soft_readout')],
            'hard_b0.35':[('hard_b0','recent_bias_with_hard_readout')],
            'soft_b0.35':[('hard_b0.35','hard_vs_soft_readout'),('soft_b0','recent_bias_with_soft_readout')]}
        for parent_view,contrast in expression_parents.get(row['view'],[]):
            if (row['job_id'],parent_view) in summary_by:
                declarations.append(dict(parent_id=row['job_id'],parent_view=parent_view,
                    contrast=contrast,gate=False,within_trained_state=True))
        for declaration in declarations:
            parentid=declaration.get('parent_id');parentjob=job_by_id.get(parentid,{})
            parent=summary_by.get((parentid,declaration.get('parent_view',row['view'])))
            if parent is None:
                parent_views=[s for s in summaries if s['job_id']==parentid]
                # Ordinary has one expression, valid as a comparator to any bank read view.
                if len(parent_views)==1 and parent_views[0]['view']=='default':parent=parent_views[0]
                # Functional byte-budget alternatives expose one ordinary readout.
                # Their declared primary bank comparator is hard selection with no
                # recent-access bias; other bank views are separately contrasted.
                if parent is None and row['view']=='default' and (
                    (job.get('memory') or {}).get('family')=='total_memory_budget' or
                    declaration.get('contrast')=='functional_equal_byte_alternative'):
                    parent=summary_by.get((parentid,'hard_b0'))
            reason=None
            if parent is None:reason='parent result or matching read view is absent'
            elif row.get('seed')!=parent.get('seed'):reason='different seeds; not a paired comparison'
            elif row.get('status')!='completed' or parent.get('status')!='completed':reason='at least one job is not completed'
            if reason:
                missing.append(dict(job_id=row['job_id'],parent_id=parentid,view=row['view'],reason=reason));continue
            contrast=declaration.get('contrast','declared_parent')
            pair=dict(panel=row['panel'],encoder=row['encoder'],encoder_variant=row['encoder_variant'],
                task=row['task'],order=row['order'],teaching=row['teaching'],seed=row['seed'],arm=row['arm'],
                parent_arm=parent['arm'],parent_encoder=parent['encoder'],parent_panel=parent['panel'],
                parent_encoder_variant=parent['encoder_variant'],parent_task=parent['task'],parent_order=parent['order'],
                parent_teaching=parent['teaching'],view=row['view'],parent_view=parent['view'],contrast=contrast,
                job_id=row['job_id'],parent_id=parentid,
                comparison_type='within_trained_state' if declaration.get('within_trained_state') else 'between_trajectories')
            mem=job.get('memory') or {}
            text=(str(contrast)+' '+str(mem.get('name'))+' '+str(mem.get('family'))).lower()
            gate=declaration.get('gate',job.get('protection_gate',any(w in text for w in ['protection','availability','release','importance','reopening'])))
            # The plasticity gate is confined to a compatible task and teacher. Encoder
            # sensitivity may have a paired contrast, but cannot certify bank protection.
            gate=bool(gate and all(row[k]==parent[k] for k in ['encoder','encoder_variant','task','order','teaching']))
            pair['protection_gate_applicable']=gate
            for key in _V24_REPORT_ENDPOINTS:
                if v24_report_number(row.get(key)) and v24_report_number(parent.get(key)):
                    pair[key+'_delta_pp']=100*(row[key]-parent[key])
            for key in ['mutable_numeric_bytes','peak_memory_numeric_bytes','peak_mutable_numeric_bytes','fixed_numeric_bytes','total_nonencoder_numeric_bytes','shared_teaching_numeric_bytes','total_learning_presentations','total_labeled_presentations']:
                if v24_report_number(row.get(key)) and v24_report_number(parent.get(key)):
                    pair[key+'_delta']=row[key]-parent[key]
            pairs.append(pair)
    grouped={}
    identity=['panel','encoder','encoder_variant','task','order','teaching','arm','parent_arm',
        'parent_encoder','parent_panel','parent_encoder_variant','parent_task','parent_order','parent_teaching','view','parent_view','contrast','comparison_type','protection_gate_applicable']
    for row in pairs:grouped.setdefault(tuple(row.get(k) for k in identity),[]).append(row)
    aggregates=[];gates=[]
    for key,rows in grouped.items():
        group=dict(zip(identity,key));group['paired_seed_count']=len(rows);group['seeds']=[r['seed'] for r in rows]
        for endpoint in _V24_REPORT_ENDPOINTS:
            stats=v24_report_interval([r.get(endpoint+'_delta_pp') for r in rows])
            for name,value in stats.items():group[endpoint+'_delta_pp_'+name]=value
        aggregates.append(group)
        if group['protection_gate_applicable']:
            acquisition=v24_report_interval([r.get('late_acquisition_delta_pp') for r in rows])
            revision=v24_report_interval([r.get('late_revision_delta_pp') for r in rows])
            enough=all(x['n']>=2 for x in [acquisition,revision])
            if enough and all(x['ci95_low']>=-5. for x in [acquisition,revision]):decision='passes_plasticity_margin'
            elif enough and any(x['ci95_high'] < -5. for x in [acquisition,revision]):decision='fails_plasticity_margin'
            else:decision='inconclusive'
            retention=v24_report_interval([r.get('late_retention_delta_pp') for r in rows])
            gates.append(dict(group,margin_pp=-5.,gate=decision,
                retention_benefit_supported=(retention['ci95_low'] is not None and retention['ci95_low']>0),
                interpretation='Plasticity margin is necessary; retained knowledge and within-bank loss must also improve. Intervals are descriptive, unadjusted screen estimates.'))
    return pairs,aggregates,gates,missing


def v24_report_aggregate(summaries):
    fields=['panel','encoder','encoder_variant','task','order','teaching','arm','family','locality','memory_locality','view']
    grouped={}
    for row in summaries:grouped.setdefault(tuple(row.get(k) for k in fields),[]).append(row)
    result=[]
    for key,rows in grouped.items():
        completed=[r for r in rows if r['status']=='completed']
        value=dict(zip(fields,key));value.update(completed_seeds=len(completed),recorded_seeds=len(rows),seeds=[r['seed'] for r in completed])
        for endpoint in _V24_REPORT_ENDPOINTS:
            stats=v24_report_interval([r.get(endpoint) for r in completed])
            for name,x in stats.items():value[endpoint+'_'+name]=x
        for costkey in ['mutable_numeric_bytes','peak_memory_numeric_bytes','peak_mutable_numeric_bytes','fixed_numeric_bytes','total_nonencoder_numeric_bytes','shared_teaching_numeric_bytes','total_learning_presentations','total_labeled_presentations']:
            value[costkey+'_mean']=v24_report_mean([r.get(costkey) for r in completed])
        result.append(value)
    return result


def v24_write_outputs(out,cfg,jobs,verification,status):
    """Aggregate completed job records. Never include raw state or checkpoints in ZIP."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    tables={k:[] for k in ['metrics','diagnostics','bank_damage','allocation_events','allocation_windows','reminders','failures','costs','representation','order_controls']}
    job_status=[];development=[];summaries=[];stream_records={}
    for job in jobs:
        path=out/'jobs'/(job['id']+'.json');meta=v24_report_job_meta(job)
        if not path.exists():
            job_status.append(dict(meta,status='not_completed',reason='No completed job record'));continue
        try:
            result=json.loads(path.read_text(encoding='utf-8'))
            if not isinstance(result,dict):raise ValueError('Job record is not an object.')
        except (ValueError,OSError) as exc:
            tables['failures'].append(dict(meta,error='Cannot read completed job record',detail=str(exc)))
            job_status.append(dict(meta,status='invalid_record',reason=str(exc)));continue
        # A result file is an atomic completed-job record; an exception/failure is
        # separately explicit and cannot become a successful completed seed.
        failed=bool(result.get('failures')) or result.get('status') in ['failed','error','incomplete']
        job_status.append(dict(meta,status='failed' if failed else 'completed',timing=result.get('timing')))
        summaries.extend(v24_report_summaries(job,result))
        for key in tables:
            data=result.get(key)
            if data is None and key=='bank_damage':data=result.get('damage')
            if isinstance(data,dict):data=[data]
            for row in data or []:
                record=dict(meta,**row) if isinstance(row,dict) else dict(meta,value=row)
                if key=='costs' and 'total_learning_presentations' not in record:
                    external=record.get('external_presentations',record.get('external_presentations_completed'))
                    if v24_report_number(external):record['total_learning_presentations']=external+record.get('extra_labeled_presentations',0)
                tables[key].append(record)
        if result.get('development'):development.append(dict(meta,development=result['development']))
        if result.get('stream'):
            value=result['stream']
            signature=json.dumps(dict(seed=job.get('seed'),task=job.get('task'),order=job.get('order'),stream=value),sort_keys=True,separators=(',',':'))
            shared=stream_records.setdefault(signature,dict(seed=job.get('seed'),task=job.get('task'),order=job.get('order'),stream=value,referenced_by_jobs=[]))
            shared['referenced_by_jobs'].append(job['id'])
    pairs,contrasts,gates,unpaired=v24_report_comparisons(jobs,summaries)
    aggregates=v24_report_aggregate(summaries)
    table_manifest=[]
    files={'metrics':'METRICS.csv','diagnostics':'DIAGNOSTICS.csv','bank_damage':'BANK_WRITE_EFFECTS.csv',
        'allocation_events':'ALLOCATION_EVENTS.csv','allocation_windows':'ALLOCATION_WINDOWS.csv',
        'reminders':'FROZEN_REMINDERS.csv','failures':'FAILURES.csv','costs':'COSTS.csv',
        'representation':'REPRESENTATION.csv','order_controls':'ORDER_CONTROLS.csv'}
    for key,name in files.items():table_manifest.append(v24_report_table(out,name,tables[key]))
    for name,rows in [('SUMMARY_BY_SEED.csv',summaries),('SUMMARY.csv',aggregates),
        ('PAIRED_COMPARISONS.csv',pairs),('PAIRED_SUMMARY.csv',contrasts),('PROTECTION_GATES.csv',gates),
        ('UNPAIRED_CONTRASTS.csv',unpaired),('JOB_STATUS.csv',job_status)]:
        table_manifest.append(v24_report_table(out,name,rows))
    coverage=v24_coverage() if 'v24_coverage' in globals() else []
    table_manifest.append(v24_report_table(out,'HYPOTHESIS_COVERAGE.csv',coverage))
    known_old=[a['name'] for a in arm_manifest()] if 'arm_manifest' in globals() else []
    coverage_text=json.dumps(v24_report_plain(coverage),sort_keys=True)
    missing_old=[name for name in known_old if ('"'+name+'"') not in coverage_text]
    cfg_plain=v24_report_plain(vars(cfg))
    source=Path(__file__).resolve() if '__file__' in globals() else None
    provenance=dict(runner='ELM V24',source_sha256=hashlib.sha256(source.read_bytes()).hexdigest() if source and source.exists() else None,
        python=sys.version,platform=platform.platform() if 'platform' in globals() else sys.platform,
        arithmetic_libraries={name:getattr(globals().get(name),'__version__',None) for name in ['np','torch']},
        source_file=source.name if source else None,
        old_arms_expected=len(known_old),old_arms_missing_coverage=missing_old,
        locality_scope='Encoder E0/E1 and historical memory references are nonlocal controls. Exact selection, normalization and scheduling remain explicitly labeled algorithmic approximations.',
        archive_scope='Summary tables, manifest, provenance and verification only; no model tensors, raw sensory inputs, raw trajectories or job-resume checkpoints.')
    completed=sum(r['status']=='completed' for r in job_status);failed=sum(r['status'] in ['failed','invalid_record'] for r in job_status)
    filters=getattr(cfg,'execution_filters',{})
    filtered=bool(cfg.smoke or cfg.max_jobs is not None or cfg.arms or cfg.no_confirmation or cfg.panels!=['all'] or set(cfg.encoders)!=set(['E0','E1','E2','E3']) or cfg.seeds!=[17,29,43,59,71,89] or cfg.confirmation_seeds!=[101,127,149,173,197,223])
    default_workload=dict(stress_waves=8,dev_cues=64,dev_passes=6,group_cues=24,variants=2,probe_variants=3,passes=3,reverse_cues=12,noise=.06,active_fraction=.25,learning_rate=.6,write_energy_floor=.5,ridge=.5)
    modified_workload={k:dict(default=v,actual=getattr(cfg,k,None)) for k,v in default_workload.items() if getattr(cfg,k,None)!=v}
    partial=filtered or bool(modified_workload) or completed!=len(jobs) or failed>0 or bool(missing_old)
    run_status=dict(requested_status=status,scope='execution_check' if cfg.smoke else ('partial_or_modified' if partial else 'full_predeclared_run'),
        planned_selected_jobs=len(jobs),completed_jobs=completed,failed_jobs=failed,
        not_completed_jobs=sum(r['status']=='not_completed' for r in job_status),
        partial=partial,filters=filters,modified_workload=modified_workload,coverage_complete=not missing_old,
        validation=verification,table_manifest=table_manifest)
    v24_report_json(out/'CONFIG.json',cfg_plain)
    v24_report_json(out/'JOB_MANIFEST.json',jobs)
    v24_report_json(out/'VERIFICATION.json',verification)
    v24_report_json(out/'PROVENANCE.json',provenance)
    v24_report_json(out/'RUN_STATUS.json',run_status)
    unique_development={}
    for entry in development:
        dev=entry['development']
        key=(entry.get('seed'),entry.get('encoder'),dev.get('encoder_sha256'))
        record=unique_development.setdefault(key,dict(seed=entry.get('seed'),encoder=entry.get('encoder'),
            development=dev,referenced_by_jobs=[]))
        record['referenced_by_jobs'].append(entry['job_id'])
    development=list(unique_development.values())
    v24_report_json(out/'DEVELOPMENT.json',development)
    def total_cost(*keys):
        values=[]
        for record in tables['costs']:
            v=next((record[k] for k in keys if v24_report_number(record.get(k))),None)
            if v is not None:values.append(v)
        return dict(total=sum(values),job_rows_with_field=len(values))
    exposure_totals=dict(
        external_trajectory_presentations=total_cost('external_presentations_completed','external_presentations'),
        extra_learning_presentations=total_cost('extra_labeled_presentations','extra_presentations'),
        total_labeled_presentations=total_cost('total_labeled_presentations'),
        total_learning_presentations=total_cost('total_learning_presentations'),
        distinct_developed_initializations=len(development),
        shared_development_training_presentations=sum(d['development'].get('development_training_presentations',0) for d in development),
        shared_development_calibration_presentations=sum(d['development'].get('development_calibration_presentations',0) for d in development),
        definition='Readout exposures sum executed trajectories. Shared development is deduplicated by seed, regime and initial-state digest. These sums are presentations, not distinct raw sensory identities across arms; re-executions after interruption are not inferred from saved checkpoints.')
    v24_report_json(out/'EXPOSURE_TOTALS.json',exposure_totals)
    v24_report_json(out/'STREAMS.json',list(stream_records.values()))
    if 'REDUCTION_PROVENANCE' in globals():v24_report_json(out/'ANATOMY_PROVENANCE.json',REDUCTION_PROVENANCE)
    lines=['# ELM V24 results','',
        '**Status: %s.** %d of %d selected jobs completed; %d failed. Requested runner status: %s.'%
        (run_status['scope'],completed,len(jobs),failed,status),'',
        'This archive reports predeclared contrasts within each encoder, task, teacher regime and read expression. '
        'There is no pooled tournament winner. A failed representative does not eliminate its mechanism family.','',
        '## Execution and coverage','',
        '- Scientific default: six comparison seeds and six fresh confirmation seeds. Filtering, altered counts and smoke mode are recorded in CONFIG.json and RUN_STATUS.json.',
        '- JOB_STATUS.csv includes absent and failed jobs. Incomplete seeds are not silently treated as successful runs.',
        '- HYPOTHESIS_COVERAGE.csv maps all 52 original arms and historical circuit ideas to execution, replacement, reference or retirement. Missing original-arm entries: %d.'%len(missing_old),
        '- Large tables are gzip-compressed CSV; the archive excludes model tensors, raw probes and resume checkpoints.',
        '- Exact cross-platform reproduction is not guaranteed. Source, library versions and stream hashes are recorded; predecessor equivalence should be checked within the same runtime environment.',
        '', '## What each comparison can establish','',
        '- E0 versus E1 tests privileged clean targets. E2 versus E3 tests local sensory development. E1 versus E3 also changes the learning objective, so it is not a pure optimizer comparison.',
        '- Circuit variants are configured after their shared E3 development and retain its frozen calibration. They test perturbations or ongoing adaptation of the same developed circuit, not independently optimized development for every variant.',
        '- Bank count, address adaptation, hard/soft reading and recent-access bias are separately labeled. Views share a trajectory only when the learner uses the selected bank\'s own error and sensory-only address adaptation.',
        '- Same-array spillover changes total write strength. Frozen-state alternatives separately match current correction and update norm; these are different diagnostics.',
        '- Fixed coefficient counts and actual byte budgets are separate controls. Equal bytes do not establish equal expressive capacity. All stored addresses, values and auxiliary numeric arrays count. Single-readout byte-budget alternatives use the bank hard_b0 view as their declared primary comparator.',
        '- Allocation windows are aligned to actual full-store events. A bounded store may forget; useful repair must admit and retain new entries through recurrence, while revision remains possible.',
        '- Same-bank current-label loss, route drift and obsolete-label recovery remain separate outcomes. Reduced score movement by itself is not memory protection.',
        '- Frozen relevant reminders must outperform unrelated reminders. Recovering a superseded answer is not successful adaptation.',
        '', '## Predeclared plasticity gate','',
        'Late acquisition and revision average the final three respective phase endpoints. A protection arm must lose no more than five percentage points on either endpoint against its declared compatible unprotected parent. '
        'PROTECTION_GATES.csv reports paired seed differences and descriptive Student 95% intervals. A pass requires both lower interval bounds at or above -5 points; a failure requires an upper bound below -5 on either endpoint; otherwise the result is inconclusive. '
        'Passing plasticity without a retention benefit is not success. Intervals are unadjusted screening intervals, not a correction for searching hundreds of configurations.',
        '', '## Panel completion','', '| Panel | Selected jobs | Completed | Failed |', '|---|---:|---:|---:|']
    for panel in sorted({str(r.get('panel')) for r in job_status}):
        rows=[r for r in job_status if str(r.get('panel'))==panel]
        lines.append('| %s | %d | %d | %d |'%(panel,len(rows),sum(r['status']=='completed' for r in rows),sum(r['status'] in ['failed','invalid_record'] for r in rows)))
    lines += ['', '## Reading the tables','',
        '- SUMMARY_BY_SEED.csv preserves every view and seed; SUMMARY.csv aggregates only matching panel/encoder/variant/task/order/teaching/arm/view groups.',
        '- PAIRED_COMPARISONS.csv retains individual paired differences, including hard/soft expression and recent-bias comparisons from each identical trained state. PAIRED_SUMMARY.csv and PROTECTION_GATES.csv aggregate only the same declared contrast.',
        '- BANK_WRITE_EFFECTS.csv, ALLOCATION_EVENTS.csv and ALLOCATION_WINDOWS.csv retain the mechanism diagnostics. REPRESENTATION.csv records code/route drift; ORDER_CONTROLS.csv records swapped-pulse controls. FROZEN_REMINDERS.csv keeps relevant, unrelated and absent reminders separate.',
        '- COSTS.csv counts external presentations, extra learning/replay and actual numeric memory. DEVELOPMENT.json deduplicates shared initializations by seed/regime/digest; EXPOSURE_TOTALS.json separates trajectory presentations from shared development. These totals are not unique examples across arms.',
        '- UNPAIRED_CONTRASTS.csv states why a declared parent could not be compared; no fabricated zero difference replaces a missing result.',
        '', 'No parameter is selected automatically from these results. Fresh-seed confirmation remains a separately labeled panel.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    archive=out.with_name(out.name+'_results.zip')
    include={t['file'] for t in table_manifest}|{'CONFIG.json','JOB_MANIFEST.json','VERIFICATION.json','PROVENANCE.json','RUN_STATUS.json','DEVELOPMENT.json','EXPOSURE_TOTALS.json','STREAMS.json','RUN_IDENTITY.json','RUNNER_ERROR.json','ANATOMY_PROVENANCE.json','REPORT.md'}
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name in sorted(include):
            p=out/name
            if p.is_file():z.write(p,p.name)
    return archive


def v24_reporting_selfcheck():
    cfg=v24_parse_config(['--smoke']);assert cfg.neurons==96 and cfg.seeds==[17] and cfg.group_cues==8
    assert cfg.confirmation_seeds==[] and cfg.no_confirmation
    stats=v24_report_interval([1.,2.,3.,4.,5.,6.]);assert stats['n']==6 and stats['ci95_low']<3.5<stats['ci95_high']
    job=dict(id='child',seed=17,panel='frozen',encoder='E3',task='association',order='original',teaching=None,
        encoder_variant={},memory=dict(name='protected',family='availability',parent='ordinary'),parent_id='base',gate=True)
    parent=dict(job,id='base',memory=dict(name='ordinary',family='control'))
    parent.pop('parent_id')
    rows=[]
    for i in range(5):
        rows += [dict(phase='novel_%02d'%i,revision=False,pass_index=1,external_presentations=2*i+1,view='default',phase_current=.8,prior_current=.7,all_current=.75),
            dict(phase='revision_%02d'%i,revision=True,pass_index=1,external_presentations=2*i+2,view='default',phase_current=.9,prior_current=.7,all_current=.75)]
    a=v24_report_summaries(job,dict(metrics=rows,costs={},failures=[]))
    b=v24_report_summaries(parent,dict(metrics=rows,costs={},failures=[]))
    assert a[0]['late_novel_phases']==3 and abs(a[0]['late_acquisition']-.8)<1e-12
    pairs,summary,gates,missing=v24_report_comparisons([job,parent],a+b)
    assert len(pairs)==1 and pairs[0]['late_revision_delta_pp']==0 and gates[0]['gate']=='inconclusive' and not missing
    bankjob=dict(parent,id='bank',memory=dict(name='bank8_fixed',family='bank'))
    bankrows=[]
    for view in ['hard_b0','soft_b0','hard_b0.35','soft_b0.35']:
        bankrows.extend(dict(row,view=view) for row in rows)
    banksummary=v24_report_summaries(bankjob,dict(metrics=bankrows,costs={},failures=[]))
    expression,_,_,_=v24_report_comparisons([bankjob],banksummary)
    assert len(expression)==4 and all(r['comparison_type']=='within_trained_state' for r in expression)
    assert all(not r['protection_gate_applicable'] for r in expression)
    budgetjob=dict(parent,id='budget',memory=dict(name='budget8_expansion',family='total_memory_budget'),
        comparisons=[dict(parent_id='bank',contrast='functional_equal_byte_alternative',gate=False)])
    budgetsummary=v24_report_summaries(budgetjob,dict(metrics=rows,costs={},failures=[]))
    allpairs,_,_,unpaired=v24_report_comparisons([bankjob,budgetjob],banksummary+budgetsummary)
    costpairs=[r for r in allpairs if r['job_id']=='budget']
    assert len(costpairs)==1 and costpairs[0]['view']=='default' and costpairs[0]['parent_view']=='hard_b0'
    assert not unpaired
    noise_rows=[dict(row,revision=False) for row in rows]
    noise_rows += [dict(phase='revise_subset',revision=False,phase_current=.99),dict(phase='revise_again',revision=False,phase_current=.99)]
    noise=v24_report_summaries(job,dict(metrics=noise_rows,costs={},failures=[]))[0]
    assert noise['late_revision'] is None and noise['late_revision_phases']==0
    assert noise['first_revision'] is None and noise['second_revision'] is None
    return dict(v24_reporting_byte_budget_primary_bank_view=True,v24_reporting_explicit_nonrevision_override=True,v24_reporting_late_phase_and_pairing=True,v24_reporting_cli_smoke=True,
        v24_reporting_single_seed_margin_inconclusive=True,v24_reporting_within_state_expression_contrasts=True)


# === V24 runtime ===
# V24 runtime, evaluation and reproducibility. No probe enters a learner update.
import copy
import traceback

V24_VERSION = '24.0.0'
V24_PARENT_SHA256 = '3723f7b7c9340f869fbdb565f3d6a08fc881e43599d991e1563a93b67a50655e'


def v24_jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {str(k): v24_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [v24_jsonable(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def v24_atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(v24_jsonable(value), separators=(',', ':'),
                               sort_keys=True, allow_nan=False)+'\n')
    temp.replace(path)


def v24_array_costs(model):
    """Count actual persistent arrays, deduplicating shared views/objects.

    Shared anatomical controller arrays are separately itemized. Python object
    allocator overhead and temporary numeric workspaces are not persistent
    information capacity and are not disguised as an exact resident-set count.
    """
    seen_objects = set()
    arrays = {}
    declared_mutable, declared_fixed = set(), set()
    def base_id(a):
        while isinstance(getattr(a, 'base', None), np.ndarray):
            a = a.base
        return id(a), a
    def walk(obj, path='', shared=False):
        if isinstance(obj, np.ndarray):
            ident, root = base_id(obj)
            item = arrays.setdefault(ident, {'path': path, 'bytes': int(root.nbytes),
                'dtype': str(root.dtype), 'shared': shared})
            item['shared'] = item['shared'] and shared
            return
        if obj is None or isinstance(obj, (str, bytes, int, float, bool, np.generic)):
            return
        if id(obj) in seen_objects:
            return
        seen_objects.add(id(obj))
        for method, dest in [('mutable_arrays', declared_mutable), ('fixed_arrays', declared_fixed)]:
            if hasattr(obj, method):
                for value in getattr(obj, method)().values():
                    if isinstance(value, np.ndarray):
                        dest.add(base_id(value)[0])
        if isinstance(obj, dict):
            items = obj.items()
        elif isinstance(obj, (list, tuple)):
            items = enumerate(obj)
        elif hasattr(obj, '__dict__'):
            items = vars(obj).items()
        else:
            return
        for key, value in items:
            walk(value, path+'.'+str(key), shared or str(key) in ('route', 'codes'))
    walk(model, 'memory')
    mutable = fixed = shared = 0
    for ident, item in arrays.items():
        if item['shared']:
            shared += item['bytes']; item['class'] = 'shared_teaching'
        elif ident in declared_fixed or ident not in declared_mutable:
            fixed += item['bytes']; item['class'] = 'fixed'
        else:
            mutable += item['bytes']; item['class'] = 'mutable'
    return {'mutable_numeric_bytes': mutable, 'fixed_numeric_bytes': fixed,
            'shared_teaching_numeric_bytes': shared,
            'total_memory_numeric_bytes': mutable+fixed,
            'array_inventory': list(arrays.values())}


class V24EventRetrievalMemory(RetrievalMemory):
    """Only consult the associative store when parametric margin is low.

    This is a causal retrieval timing rule. It does not alter the local fallback
    update or silently count skipped lookups as performed work.
    """
    def __init__(self, d, seed, trigger_margin=.3, **params):
        super().__init__(d, seed, **params)
        self.trigger_margin = float(trigger_margin)
        self.triggered_lookups = 0

    def output(self, H):
        H = np.asarray(H); matrix = np.atleast_2d(H)
        base = matrix@self.V
        ordered = np.sort(base, axis=1)
        selected = ordered[:, -1]-ordered[:, -2] < self.trigger_margin
        out = base.copy()
        if np.any(selected):
            retrieved, hits, _ = self._lookup(matrix[selected])
            alpha = self.mix*hits[:, None]
            out[selected] = (1-alpha)*base[selected]+alpha*retrieved
        return out[0] if H.ndim == 1 else out

    def observe(self, h, label):
        base = np.asarray(h)@self.V; ordered = np.sort(base)
        if ordered[-1]-ordered[-2] < self.trigger_margin:
            _, hits, best = self._lookup(h)
            self.triggered_lookups += 1
            self.online_lookup_hits += int(hits[0])
            self.online_lookup_similarity_sum += float(best[0])
        change = delta_change(h, target4(label)-base, self.eta)
        self.V += change
        self._insert_observed(h, label); self.t += 1
        return {'extra_labeled_presentations': 0, 'update_norm': float(np.linalg.norm(change)),
                'lookup_triggered': int(ordered[-1]-ordered[-2] < self.trigger_margin)}


class V24CompetitionMemory(MemoryBase):
    """Fixed four-output recurrent inhibition, with local teaching feedback.

    Applies to ordinary/unprotected full-width banks only. There is no reverse
    derivative through competition. Its output is a transmitted activity, and
    target minus activity is delivered directly to the selected readout.
    """
    def __init__(self, inner, strength=.2):
        super().__init__(inner.d, 0)
        self.inner = inner; self.strength = float(strength)
        self.competition = np.ones((4, 4))-np.eye(4)
        self.t = inner.t

    def _compete(self, z):
        z = np.asarray(z); a = np.maximum(z, 0)
        for _ in range(4):
            a = .5*a+.5*np.maximum(z-self.strength*(a@self.competition), 0)
        return a

    def output(self, H):
        return self._compete(self.inner.output(H))

    def output_view(self, H, read_mode='hard', recent_bias=0., reminders=None):
        return self._compete(self.inner.output_view(H, read_mode, recent_bias, reminders))

    def observe(self, h, label):
        bank = isinstance(self.inner, V24BankMemory)
        j = int(np.argmax(self.inner.route_probabilities(h))) if bank else None
        V = self.inner.V.copy()
        raw = h@(V[j] if bank else V)
        extra = delta_change(h, raw-self._compete(raw), self.inner.eta)
        info = self.inner.observe(h, label)
        if bank:
            values = self.inner.V.copy(); values[j] += extra; self.inner.V = values
        else:
            self.inner.V += extra
        self.t = self.inner.t
        info['competition_strength'] = self.strength
        info['update_norm'] = float(np.linalg.norm(self.inner.V-V))
        return info

    def mutable_arrays(self):
        return {'inner.'+k:v for k,v in self.inner.mutable_arrays().items()}

    def fixed_arrays(self):
        return {'competition':self.competition}

    def diagnostics(self):
        return dict(self.inner.diagnostics(), competition_strength=self.strength)

    def state_digest(self):
        return sha_bytes((self.inner.state_digest()+str(self.strength)).encode())


class V24PrecisionMemory(MemoryBase):
    """Actual stored float32/int8 readout weights, with transient dequantization.

    Fixed +/-4 bound and uniform int8 grid; no scale learned from future weights.
    Other synaptic/route traces retain their declared precision and are counted.
    """
    def __init__(self, inner, precision):
        super().__init__(inner.d, 0)
        self.inner = inner; self.precision = precision; self.bound = 4.
        self.key = '_V' if hasattr(inner, '_V') else 'V'
        initial = np.asarray(getattr(inner, self.key))
        self.packed = (np.rint(np.clip(initial/self.bound, -1, 1)*127).astype(np.int8)
                       if precision == 'quantized8' else initial.astype(np.float32))
        setattr(inner, self.key, None)
        self.t = inner.t

    def _call(self, method, *args, write=False, **kwargs):
        values = (self.packed.astype(np.float64)*(self.bound/127.)
                  if self.precision == 'quantized8' else self.packed.copy())
        setattr(self.inner, self.key, values)
        try:
            result = getattr(self.inner, method)(*args, **kwargs)
            if write:
                values = np.asarray(getattr(self.inner, self.key))
                self.packed = (np.rint(np.clip(values/self.bound, -1, 1)*127).astype(np.int8)
                               if self.precision == 'quantized8' else values.astype(np.float32))
                self.t = self.inner.t
            return result
        finally:
            setattr(self.inner, self.key, None)

    def output(self, H):
        return self._call('output', H)

    def output_view(self, H, read_mode='hard', recent_bias=0., reminders=None):
        return self._call('output_view', H, read_mode, recent_bias, reminders)

    def observe(self, h, label):
        before=self.packed.copy()
        info=self._call('observe', h, label, write=True)
        scale=self.bound/127. if self.precision=='quantized8' else 1.
        info['update_norm']=float(np.linalg.norm((self.packed.astype(float)-before.astype(float))*scale))
        return info

    def mutable_arrays(self):
        result = {k:v for k,v in self.inner.mutable_arrays().items() if isinstance(v,np.ndarray)}
        return dict(result, packed_weights=self.packed)

    def fixed_arrays(self):
        return self.inner.fixed_arrays() if hasattr(self.inner,'fixed_arrays') else {}

    def diagnostics(self):
        # Some diagnostics inspect V; materialize without retaining it.
        return dict(self._call('diagnostics'), stored_weight_precision=self.precision,
                    stored_weight_bytes=int(self.packed.nbytes))

    def state_digest(self):
        return sha_bytes(self.packed.tobytes()+self.inner.state_digest().encode())


def v24_make_memory(spec, seed, route, codes, cfg):
    s = copy.deepcopy(spec); p = s.setdefault('params', {})
    competition = p.pop('readout_competition', 0.)
    precision = p.pop('memory_precision', None)
    timing = p.pop('retrieval_timing', 'continuous')
    # Paired new bank and allocation variants start from identical family draws.
    if s['kind'] == 'bank24':
        model = V24BankMemory(96, seed_for(seed,'v24_banks'), **p)
    elif s['kind'] == 'allocation24':
        model = V24AllocationMemory(96, seed_for(seed,'v24_allocation'), **p)
    elif s['kind'] == 'turnover24':
        model = V24LocalTurnoverMemory(96, seed_for(seed,'v24_turnover'), **p)
    elif s['kind'] == 'expansion24':
        model = V24ExpansionMemory(96, seed_for(seed,'v24_expansion'), **p)
    elif s['kind'] == 'budget24':
        model = v24_budget_memory(96, seed_for(seed,'v24_budget'),
                                 byte_cost=lambda m:v24_array_costs(m)['total_memory_numeric_bytes'],**p)
    elif s['kind'] in ('context_gating','silent_memory'):
        model=V24LegacyBankMemory(96,seed_for(seed,'v23_family_'+s['family']),
                                 recent_bias=0. if s['kind']=='context_gating' else .35,**p)
    elif s['name'] in ('ordinary', 'ordinary_slow') or (
            s['kind']=='legacy' and p.get('mode')=='uniform_delta'):
        model = PlainMemory(96, seed_for(seed,'v23_family_reference'),
                            p.get('eta', cfg.learning_rate))
    elif s['kind']=='retrieval' and timing=='event':
        model = V24EventRetrievalMemory(96,seed_for(seed,'v23_family_'+s['family']),**p)
    else:
        model = make_memory(s, seed, route, codes, cfg)
    if competition:
        model = V24CompetitionMemory(model, competition)
    if precision:
        if competition or not isinstance(model,(PlainMemory,V24BankMemory)):
            raise ValueError('Precision wrapper supports plain or bank readouts without competition.')
        model = V24PrecisionMemory(model, precision)
    return model


def v24_underlying_bank(model):
    if isinstance(model, (V24BankMemory, ContentBankMemory)):
        return model
    if isinstance(model, V24CompetitionMemory):
        return v24_underlying_bank(model.inner)
    return None  # Quantized banks require transient storage; generic evaluator remains valid.


def v24_is_bank(model):
    if isinstance(model, V24PrecisionMemory):
        return v24_is_bank(model.inner)
    return v24_underlying_bank(model) is not None


def v24_bank_probabilities(bank, H, read_mode='hard'):
    if isinstance(bank, V24BankMemory):
        return np.atleast_2d(bank.route_probabilities(H, read_mode=read_mode))
    H = np.atleast_2d(unit_rows(H)); out = np.zeros((len(H), bank.banks))
    if bank.routing=='fixed':
        logits = H@bank.fixed_address.T; used=bank.banks
    elif bank.used:
        logits = H@bank.centers[:bank.used].T; used=bank.used
    else:
        return out
    if read_mode=='hard':
        out[np.arange(len(H)),np.argmax(logits,axis=1)] = 1.
    else:
        out[:,:used] = _structural_softmax(bank.temperature*logits)
    return out


def v24_scores(model, H, view='default', reminders=None):
    if view=='default':
        return np.asarray(model.output(H))
    read_mode, beta_text = view.split('_b')
    beta=float(beta_text)
    if isinstance(model,(V24BankMemory,V24PrecisionMemory,V24CompetitionMemory)):
        return np.asarray(model.output_view(H,read_mode,beta,reminders))
    bank=v24_underlying_bank(model)
    if bank is None:
        raise ValueError('Expression view requested for non-bank memory.')
    matrix=np.atleast_2d(H); p=v24_bank_probabilities(bank,matrix,read_mode)
    prior=np.zeros(bank.banks)
    if reminders is not None and len(reminders):
        prior=v24_bank_probabilities(bank,reminders,read_mode).mean(axis=0)
    elif bank.active>=0:
        prior[bank.active]=1.
    if prior.sum(): p=(1-beta)*p+beta*prior
    z=np.sum(p[:,:,None]*np.einsum('nd,bdc->nbc',matrix,bank.V,optimize=False),axis=1)
    return z[0] if np.asarray(H).ndim==1 else z


def v24_views(model):
    return ['hard_b0','soft_b0','hard_b0.35','soft_b0.35'] if v24_is_bank(model) else ['default']


def v24_metric_scores(z, ph, stream, cfg, seen=None):
    N=len(stream['original']); variants=cfg.probe_variants
    z=np.asarray(z).reshape(N,variants,4)
    seen=ph['seen'] if seen is None else seen
    ids=np.zeros(N,bool);ids[ph['ids']]=True
    a=np.arange(N)<cfg.group_cues
    b=(np.arange(N)>=cfg.group_cues)&(np.arange(N)<2*cfg.group_cues)
    c=(np.arange(N)>=2*cfg.group_cues)&(np.arange(N)<3*cfg.group_cues)
    masks=dict(all_current=seen,A_current=a&seen,B_current=b&seen,C_current=c&seen,
        initial_revised=stream['initial_reverse']&seen,
        original_never_revised=seen&~ph['ever_revised'],phase_current=ids&seen,
        prior_current=seen&~ids,initial_unchanged=(a|b)&~stream['initial_reverse']&seen)
    ans={name:accuracy(z,ph['targets'],mask,variants) for name,mask in masks.items()}
    if np.any(seen):
        target=np.eye(4)[ph['targets'][seen]][:,None,:]
        ans['current_mse']=float(np.mean((z[seen]-target)**2))
    else: ans['current_mse']=None
    ans['seen_identities']=int(np.sum(seen))
    return ans


def v24_loss(z, targets):
    return np.mean((np.asarray(z)-np.eye(4)[np.asarray(targets)])**2,axis=-1)


def v24_snapshot_predictions(H, V, p, active=-1, beta=0.):
    p=p.copy()
    if beta and active>=0:
        p*=1-beta;p[:,active]+=beta
    return np.sum(p[:,:,None]*np.einsum('nd,bdc->nbc',H,V,optimize=False),axis=1)


def v24_bank_damage_before(model, h, H):
    bank=v24_underlying_bank(model)
    if bank is None or isinstance(model,V24CompetitionMemory): return None
    return {'V':bank.V.copy(), 'p':v24_bank_probabilities(bank,H,'hard'),
            'soft':v24_bank_probabilities(bank,H,'soft'),
            'current_p':v24_bank_probabilities(bank,h,'soft')[0],
            'active':bank.active,'h':h.copy(),'H':H}


def v24_bank_damage_after(model, before, label, selected_bank, targets, last_routes):
    bank=v24_underlying_bank(model); H=before['H']; h=before['h']; V0=before['V'];V1=bank.V
    j=int(selected_bank); p0=before['p']; p1=v24_bank_probabilities(bank,H,'hard')
    z0=v24_snapshot_predictions(H,V0,p0)
    storage=v24_snapshot_predictions(H,V1,p0)
    rerouted=v24_snapshot_predictions(H,V1,p1)
    prior0=v24_snapshot_predictions(H,V1,p1,before['active'],.35)
    prior1=v24_snapshot_predictions(H,V1,p1,bank.active,.35)
    same=np.argmax(p0,axis=1)==j
    rows=[]
    for name,mask in [('same_bank',same),('other_bank',~same)]:
        if not mask.any(): continue
        losses0=v24_loss(z0[mask],targets[mask])
        rows.append(dict(group=name,probe_count=int(mask.sum()),
            storage_mse_change=float(np.mean(v24_loss(storage[mask],targets[mask])-losses0)),
            storage_score_displacement=float(np.mean((storage[mask]-z0[mask])**2)),
            router_mse_change=float(np.mean(v24_loss(rerouted[mask],targets[mask])-v24_loss(storage[mask],targets[mask]))),
            access_prior_mse_change=float(np.mean(v24_loss(prior1[mask],targets[mask])-v24_loss(prior0[mask],targets[mask]))),
            soft_route_overlap=float(np.mean(before['soft'][mask,j])),
            route_changed_fraction=float(np.mean(np.argmax(p0[mask],axis=1)!=np.argmax(p1[mask],axis=1))),
            previous_learning_route_agreement=float(np.mean(last_routes[mask]==np.argmax(p0[mask],axis=1)))
                if np.all(last_routes[mask]>=0) else None))
    error=target4(label)-h@V0[j]
    if getattr(bank,'coefficient_budget',None) is not None:
        # Actual compact writes above are valid. Unrestricted alternate writes
        # would introduce forbidden feature connections, so do not compare them.
        return rows
    reference=delta_change(h,error,getattr(bank,'eta',.6))
    actual=V1-V0; p=before['current_p']
    for allocation_name,a in [('isolated',np.eye(bank.banks)[j]),
                              ('spillover',np.full(bank.banks,.1/max(1,bank.banks-1))),
                              ('uniform',np.ones(bank.banks))]:
        if allocation_name=='spillover':a[j]=1.
        denominator=float(p@a)
        if denominator<=1e-12 or p[j]<=1e-12:continue
        change=(p[j]*a/denominator)[:,None,None]*reference[None,:,:]
        counter=v24_snapshot_predictions(H,V0+change,before['soft'])
        base=v24_snapshot_predictions(H,V0,before['soft'])
        corrected=np.sum(p[:,None]*(h@change),axis=0)
        desired=p[j]*(h@reference)
        rows.append(dict(group='matched_current_soft_'+allocation_name,
            probe_count=len(H),old_mse_change=float(np.mean(v24_loss(counter,targets)-v24_loss(base,targets))),
            update_norm=float(np.linalg.norm(change)),actual_update_norm=float(np.linalg.norm(actual)),
            correction_match_error=float(np.linalg.norm(corrected-desired))))
        norm=float(np.linalg.norm(change));actual_norm=float(np.linalg.norm(actual))
        if norm>1e-12:
            scaled=change*(actual_norm/norm)
            alternate=v24_snapshot_predictions(H,V0+scaled,before['soft'])
            rows.append(dict(group='matched_write_norm_'+allocation_name,probe_count=len(H),
                old_mse_change=float(np.mean(v24_loss(alternate,targets)-v24_loss(base,targets))),
                update_norm=float(np.linalg.norm(scaled)),actual_update_norm=actual_norm,
                current_correction_fraction=float(np.linalg.norm(np.sum(p[:,None]*(h@scaled),axis=0))/max(np.linalg.norm(desired),1e-12))))
    return rows


def v24_reminders(model,H,ph,stream,cfg):
    if not v24_is_bank(model): return []
    seen=ph['seen'];a=(np.arange(len(seen))<cfg.group_cues)&seen
    other=np.flatnonzero(seen&~a)
    if not a.any() or not len(other):return []
    reminder_ids=np.flatnonzero(a)[:min(3,max(1,int(a.sum())//2))]
    score_ids=np.flatnonzero(a&~np.isin(np.arange(len(a)),reminder_ids))
    if not len(score_ids):return []
    targets=np.repeat(ph['targets'][score_ids],cfg.probe_variants)
    probe=H[score_ids].reshape(-1,96);original=np.repeat(stream['original'][score_ids],cfg.probe_variants)
    relevant=H[reminder_ids,0]; unrelated=H[other[:len(reminder_ids)],0]
    digest=model.state_digest(); rows=[]
    for view in v24_views(model):
        for condition,reminder in [('none',None),('relevant',relevant),('unrelated',unrelated)]:
            scores=v24_scores(model,probe,view,reminder)
            changed=targets!=original
            rows.append(dict(view=view,condition=condition,
                current_accuracy=float(np.mean(scores.argmax(-1)==targets)),
                obsolete_accuracy=float(np.mean(scores[changed].argmax(-1)==original[changed])) if changed.any() else None,
                probe_count=len(probe),reminder_count=0 if reminder is None else len(reminder),
                scored_reminder_identity_overlap=0))
    if digest!=model.state_digest():raise AssertionError('Reminder changed stored state.')
    return rows


def v24_cached_features(encoder,stream,spec,cfg):
    feature=spec.get('feature','sparse'); result={}
    for key in ('train','probe'):
        x=stream[key]; shape=x.shape[:2]
        cut=min(len(x),2*cfg.group_cues)
        pieces=[encoder.features(x[:cut].reshape(-1,12,32),feature=feature,adapt=False)]
        if cut<len(x):pieces.append(encoder.features(x[cut:].reshape(-1,12,32),feature=feature,adapt=False))
        result[key]=np.concatenate(pieces).reshape(*shape,96)
    return result


def v24_run_standard_job(cfg,job,encoder,stream,model,cache=None):
    start=time.perf_counter(); feature=job['memory'].get('feature','sparse')
    live=bool(job.get('encoder_variant'))
    # Caching is only used for invariant encoders and unchanged sensory inputs.
    if not live and cache is None:cache=v24_cached_features(encoder,stream,job['memory'],cfg)
    result={name:[] for name in ('metrics','diagnostics','bank_damage','allocation_events',
              'allocation_windows','reminders','failures','representation','order_controls')}
    N=stream['unique_identities']; seen=np.zeros(N,bool); last_routes=np.full(N,-1,int)
    previous_probe_features=None; previous_probe_routes=None
    external=extra=0;first_full=None;allocation_pending={};allocation_admission_counts={}
    slot_id={};slot_last_identity={};slot_birth={};slot_last={};slot_id_sets={};admission_serial=0
    windows={0,1,8,32,64,128}; window_done=set()
    peak=v24_array_costs(model)['total_memory_numeric_bytes']
    completed=False; eval_seconds=0.; update_seconds=0.;encoder_online_events=0
    for ph in stream['schedule']:
        phase_stats={}; phase_samples=0
        override=ph.get('train_override')
        override_feat=None
        if override is not None and not live:
            override_feat=encoder.features(override.reshape(-1,12,32),feature=feature,adapt=False).reshape(*override.shape[:2],96)
        for cp in ph['checkpoints']:
            correct=0; count=0
            for identity,variant,label in cp['events']:
                external+=1
                try:
                    tick=time.perf_counter()
                    if live:
                        raw=(override if override is not None else stream['train'])[identity,variant]
                        h=np.asarray(encoder.features(raw,feature=feature,adapt=True)).reshape(96)
                        encoder_online_events+=1
                    else:
                        h=(override_feat if override_feat is not None else cache['train'])[identity,variant]
                    primary_view='hard_b0' if v24_is_bank(model) else 'default'
                    before_scores=v24_scores(model,h,primary_view)
                    if before_scores.shape!=(4,) or not np.isfinite(before_scores).all():
                        raise FloatingPointError('Invalid prequential prediction.')
                    correct+=int(np.argmax(before_scores)==label);count+=1
                    # Remember first external recurrence after admission, including eviction.
                    for admission in allocation_pending.pop(identity,[]):
                        result['allocation_events'].append(dict(event='first_recurrence',external=external,
                            admission=admission['serial'],identity=identity,slot=admission['slot'],
                            age=external-admission['time'],
                            admission_generation_survives=int(model.generation[admission['slot']])==admission['generation'],
                            prequential_correct=int(np.argmax(before_scores)==label)))
                    eligible=seen.copy();eligible[identity]=False
                    if ph['revision']:eligible[np.asarray(ph['ids'],int)]=False
                    ids=np.flatnonzero(eligible)
                    measure=bool(cfg.diagnostic_interval and external%cfg.diagnostic_interval==0 and len(ids))
                    selected=ids[:0];H=None;bank_before=None;old_scores=None
                    if measure:
                        shift=(external//cfg.diagnostic_interval)%len(ids)
                        selected=np.roll(ids,-shift)[:8]
                        H=(encoder.features(stream['probe'][selected].reshape(-1,12,32),feature=feature,adapt=False)
                           if live else cache['probe'][selected].reshape(-1,96))
                        old_scores=np.asarray(model.output(H))
                        bank_before=v24_bank_damage_before(model,h,H)
                    allocation_before=copy.deepcopy(model) if measure and isinstance(model,V24AllocationMemory) else None
                    info=model.observe(h,label) or {}
                    if model.t!=external:raise AssertionError('External event clock mismatch.')
                    new_extra=info.get('extra_labeled_presentations',0)
                    if new_extra<0 or int(new_extra)!=new_extra:raise AssertionError('Invalid extra presentation count.')
                    extra+=int(new_extra);aggregate_info(phase_stats,info)
                    after_scores=v24_scores(model,h,primary_view)
                    err=target4(label)-before_scores
                    phase_stats.setdefault('achieved_correction_fraction',[0.,0])
                    phase_stats['achieved_correction_fraction'][0]+=float(np.linalg.norm(after_scores-before_scores)/max(np.linalg.norm(err),1e-12))
                    phase_stats['achieved_correction_fraction'][1]+=1
                    if measure:
                        targets=np.repeat(ph['targets'][selected],cfg.probe_variants)
                        if bank_before is not None:
                            j=int(info.get('selected_bank',np.argmax(bank_before['current_p'])))
                            rows=v24_bank_damage_after(model,bank_before,label,j,targets,
                                                       np.repeat(last_routes[selected],cfg.probe_variants))
                            for row in rows:result['bank_damage'].append(dict(phase=ph['name'],external=external,**row))
                        else:
                            after=np.asarray(model.output(H))
                            result['bank_damage'].append(dict(phase=ph['name'],external=external,group='generic_memory',
                                probe_count=len(H),old_mse_change=float(np.mean(v24_loss(after,targets)-v24_loss(old_scores,targets))),
                                score_displacement=float(np.mean((after-old_scores)**2))))
                    if 'selected_bank' in info:last_routes[identity]=int(info['selected_bank'])
                    if isinstance(model,V24AllocationMemory):
                        slot=int(info.get('selected_slot',-1));allocated=bool(info.get('allocated',0) or info.get('recycled',0))
                        if slot>=0:
                            victim=slot_id.get(slot,-1)
                            if allocated:
                                admission_serial+=1
                                allocation_admission_counts[identity]=allocation_admission_counts.get(identity,0)+1
                                row=dict(event='replacement' if info.get('recycled',0) else 'admission',external=external,
                                    admission=admission_serial,identity=identity,slot=slot,victim_identity=victim,
                                    victim_age=external-slot_birth[slot] if slot in slot_birth else None,
                                    victim_last_hit_age=external-slot_last[slot] if slot in slot_last else None,
                                    identity_admission_count=allocation_admission_counts[identity],
                                    actual_input_correct_after_first_write=int(np.argmax(after_scores)==label))
                                if allocation_before is not None and victim>=0:
                                    victimH=(encoder.features(stream['probe'][victim],feature=feature,adapt=False)
                                             if live else cache['probe'][victim])
                                    victimtargets=np.full(cfg.probe_variants,ph['targets'][victim],int)
                                    row['sampled_victim_mse_change']=float(np.mean(v24_loss(model.output(victimH),victimtargets)
                                        -v24_loss(allocation_before.output(victimH),victimtargets)))
                                result['allocation_events'].append(row)
                                allocation_pending.setdefault(identity,[]).append(dict(serial=admission_serial,slot=slot,time=external,
                                    generation=int(model.generation[slot])))
                                slot_birth[slot]=external;slot_id_sets[slot]=set()
                                slot_id[slot]=identity
                            slot_last_identity[slot]=identity;slot_last[slot]=external
                            slot_id_sets.setdefault(slot,set()).add(identity)
                        if first_full is None and model.used>=model.slots:first_full=external
                    seen[identity]=True
                    update_seconds+=time.perf_counter()-tick
                    if first_full is not None and external-first_full in windows and external-first_full not in window_done:
                        offset=external-first_full;window_done.add(offset);tick=time.perf_counter()
                        hp=(encoder.features(stream['probe'].reshape(-1,12,32),feature=feature,adapt=False).reshape(N,cfg.probe_variants,96)
                            if live else cache['probe'])
                        m=v24_metric_scores(model.output(hp.reshape(-1,96)),ph,stream,cfg,seen)
                        result['allocation_windows'].append(dict(phase=ph['name'],external=external,offset_from_first_full=offset,
                            first_full_external=first_full,occupied_slots=model.used,
                            evaluator_admitted_owner_identities=len(set(slot_id.values())),
                            evaluator_contacted_identities=len(set().union(*slot_id_sets.values())) if slot_id_sets else 0,
                            evaluator_duplicate_slot_count=len(slot_id)-len(set(slot_id.values())),
                            evaluator_cross_identity_slots=sum(len(v)>1 for v in slot_id_sets.values()),**m))
                        eval_seconds+=time.perf_counter()-tick
                except Exception as exc:
                    result['failures'].append(dict(phase=ph['name'],external=external,
                        exception=type(exc).__name__,message=str(exc),traceback=traceback.format_exc(limit=8)))
                    break
            if result['failures']:break
            tick=time.perf_counter();digest=model.state_digest();encoder_digest=encoder.state_digest()
            hp=(encoder.features(stream['probe'].reshape(-1,12,32),feature=feature,adapt=False).reshape(N,cfg.probe_variants,96)
                if live else cache['probe'])
            for view in v24_views(model):
                z=v24_scores(model,hp.reshape(-1,96),view)
                if z.shape!=(N*cfg.probe_variants,4) or not np.isfinite(z).all():
                    raise FloatingPointError('Invalid checkpoint output.')
                result['metrics'].append(dict(phase=ph['name'],revision=ph['revision'],pass_index=cp['pass_index'],
                    view=view,external_presentations=external,extra_labeled_presentations=extra,
                    total_labeled_presentations=external+extra,
                    prequential_accuracy=correct/max(1,count) if view in ('default','hard_b0') else None,
                    **v24_metric_scores(z,ph,stream,cfg)))
            if digest!=model.state_digest():raise AssertionError('Evaluation mutated memory.')
            if encoder_digest!=encoder.state_digest():raise AssertionError('Evaluation mutated encoder.')
            eval_seconds+=time.perf_counter()-tick
        if result['failures']:break
        tick=time.perf_counter()
        numeric={k:v for k,v in model.diagnostics().items() if isinstance(v,(int,float,bool,str,np.generic))}
        avg={'write_mean_'+k:v[0]/v[1] for k,v in phase_stats.items() if v[1]}
        result['diagnostics'].append(dict(phase=ph['name'],external=external,**numeric,**avg))
        if previous_probe_features is not None:
            result['representation'].append(dict(phase=ph['name'],external=external,
                feature_mse_drift=float(np.mean((hp-previous_probe_features)**2)),
                active_fraction=float(np.mean(hp!=0))))
        previous_probe_features=hp.copy()
        bank=v24_underlying_bank(model)
        if bank is not None:
            routes=np.argmax(v24_bank_probabilities(bank,hp.reshape(-1,96)),axis=1)
            if previous_probe_routes is not None:
                result['representation'].append(dict(phase=ph['name'],external=external,
                    same_cue_route_change_fraction=float(np.mean(routes!=previous_probe_routes))))
            previous_probe_routes=routes.copy()
        if ph['name'] in ('interfere_B','interfere_C','revise_again') or ph is stream['schedule'][-1]:
            for row in v24_reminders(model,hp,ph,stream,cfg):
                result['reminders'].append(dict(phase=ph['name'],external=external,**row))
        if 'structured_controls' in stream and ph is stream['schedule'][-1]:
            controls=stream['structured_controls']
            raw=controls['probe_swapped']
            swapped=encoder.features(raw.reshape(-1,12,32),feature=feature,adapt=False)
            sensitive=np.asarray(controls['order_sensitive_mask'])&ph['seen']
            for view in v24_views(model):
                scores=v24_scores(model,swapped,view)
                result['order_controls'].append(dict(phase=ph['name'],view=view,
                    swapped_target_accuracy=accuracy(scores,controls['targets_swapped'],ph['seen'],cfg.probe_variants),
                    order_sensitive_accuracy=accuracy(scores,controls['targets_swapped'],sensitive,cfg.probe_variants),
                    unchanged_answer_fraction=accuracy(scores,ph['targets'],sensitive,cfg.probe_variants)))
        peak=max(peak,v24_array_costs(model)['total_memory_numeric_bytes'])
        eval_seconds+=time.perf_counter()-tick
    if not result['failures']:completed=True
    result['costs']=dict(v24_array_costs(model),peak_memory_numeric_bytes=peak,
        external_presentations_completed=model.t,extra_labeled_presentations=extra,
        total_labeled_presentations=model.t+extra,online_encoder_events=encoder_online_events,
        first_full_external=first_full,status='completed' if completed else 'failed',
        final_state_sha256=model.state_digest(),encoder_costs=encoder.costs())
    result['costs']['total_learning_presentations']=model.t+extra
    result['costs']['external_readout_update_calls']=model.t
    result['costs']['primary_prequential_prediction_calls']=model.t
    result['costs']['deployment_sensory_forwards']=model.t
    for key in ('v24_target_array_bytes','v24_unused_array_bytes','v24_budget_target_banks','v24_budget_basis'):
        if hasattr(model,key):result['costs'][key]=getattr(model,key)
    result['timing']=dict(seconds=time.perf_counter()-start,update_and_sample_seconds=update_seconds,
                          checkpoint_evaluator_seconds=eval_seconds)
    return result


def v24_runtime_checks(cfg,anatomy):
    rg=np.random.default_rng(7341);H=unit_rows(rg.normal(size=(8,96)))
    codes=make_codebook(17);route=Route(anatomy,17,codes)
    names=[]
    for spec in v24_memory_specs():
        model=v24_make_memory(spec,17,route,codes,cfg)
        inputs_before=H.copy()
        for i,h in enumerate(H):
            model.observe(h,i%4)
        if not np.array_equal(inputs_before,H):raise AssertionError('Memory changed shared feature inputs: '+spec['name'])
        if model.t!=len(H):raise AssertionError('Memory clock: '+spec['name'])
        before=model.state_digest()
        for view in v24_views(model):
            z=v24_scores(model,H,view)
            if z.shape!=(8,4) or not np.isfinite(z).all():raise AssertionError(spec['name'])
        if before!=model.state_digest():raise AssertionError('Impure memory: '+spec['name'])
        cost=v24_array_costs(model)
        if cost['total_memory_numeric_bytes']<=0:raise AssertionError('Empty cost ledger')
        if spec['kind']=='budget24':
            limit=getattr(model,'v24_target_array_bytes',None)
            if limit is not None and cost['total_memory_numeric_bytes']>limit:
                raise AssertionError('Memory budget exceeded')
        names.append(spec['name'])
    # New scalar delta baseline is numerically the factored archived update.
    old=LegacyMemory(96,17,route,codes,cfg,'uniform_delta');new=PlainMemory(96,17,cfg.learning_rate)
    for i,h in enumerate(H):old.observe(h,i%4);new.observe(h,i%4)
    error=float(np.max(np.abs(old.output(H)-new.output(H))))
    if error>1e-12:raise AssertionError('Ordinary reference equivalence failed')
    for banked in (False,True):
        for precision in ('float32','quantized8'):
            base=V24BankMemory(96,17,banks=8) if banked else PlainMemory(96,17)
            model=V24PrecisionMemory(base,precision)
            for i,h in enumerate(H):model.observe(h,i%4)
            digest=model.state_digest();out=model.output(H)
            if not np.isfinite(out).all() or digest!=model.state_digest():raise AssertionError('Precision purity')
            assert model.packed.dtype==(np.int8 if precision=='quantized8' else np.float32)
    return dict(all_memory_specs_checked=len(names),ordinary_reference_max_error=error,
                actual_weight_precision_checked=True,all_prediction_views_pure=True)


def v24_main():
    cfg=v24_parse_config();jobs=v24_jobs(cfg)
    if cfg.list_jobs:
        for job in jobs:
            print(json.dumps(job,sort_keys=True))
        print('Selected jobs: %d'%len(jobs),file=sys.stderr)
        return 0
    if cfg.list_arms:
        unique={j['memory']['name']:j['memory'] for j in jobs}
        for name,spec in unique.items():print(name+' '+json.dumps(spec,sort_keys=True))
        return 0
    if not jobs:raise SystemExit('No jobs match the requested selection.')
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
        os.environ[key]=str(cfg.threads)
    dependencies(cfg.install_deps);torch.set_num_threads(cfg.threads)
    try:torch.set_num_interop_threads(1)
    except RuntimeError:pass
    torch.use_deterministic_algorithms(True)
    anatomy=load_anatomy()
    print('ELM V24: %d selected trajectories; four pure bank read views where applicable.'%len(jobs),flush=True)
    verification=dict(manifest=v24_manifest_selfcheck(),memory=v24_memory_checks(),
        encoder=v24_encoder_checks(),teaching=v24_teaching_checks(),
        reporting=v24_reporting_selfcheck(),runtime=v24_runtime_checks(cfg,anatomy))
    print('Arithmetic, locality-interface, memory-budget and pure-evaluation checks passed.',flush=True)
    if cfg.self_test:
        print(json.dumps(v24_jsonable(verification),indent=2));return 0
    source_hash=sha_bytes(Path(__file__).read_bytes())
    args=v24_jsonable(vars(cfg))
    identity={k:v for k,v in args.items() if k not in ('resume','out_dir','max_jobs',
        'install_deps','list_jobs','list_arms','self_test','execution_filters')}
    identity['source_sha256']=source_hash
    identity['jobs']=[j['id'] for j in jobs]
    fingerprint=sha_bytes(json.dumps(identity,sort_keys=True).encode())
    out=(cfg.resume or cfg.out_dir or Path('elm_fly_v24_'+time.strftime('%Y%m%d_%H%M%S')+'_'+str(os.getpid()))).expanduser().resolve()
    if cfg.resume:
        prior=json.loads((out/'RUN_IDENTITY.json').read_text())
        if prior.get('fingerprint')!=fingerprint:
            raise SystemExit('Resume source/configuration differs. Use the original script and experiment arguments.')
    else:out.mkdir(parents=True,exist_ok=False)
    (out/'jobs').mkdir(exist_ok=True)
    v24_atomic_json(out/'RUN_IDENTITY.json',dict(fingerprint=fingerprint,identity=identity,
        source_sha256=source_hash,version=V24_VERSION))
    v24_atomic_json(out/'JOB_MANIFEST.json',jobs)
    v24_atomic_json(out/'CONFIG.json',args)
    v24_atomic_json(out/'VERIFICATION.json',verification)
    print('Output: '+str(out),flush=True)
    print('Default association stream: %d external presentations per trajectory.'%
        ((3*cfg.group_cues+2*cfg.reverse_cues+cfg.stress_waves*(cfg.group_cues+cfg.reverse_cues))*cfg.variants*cfg.passes),flush=True)
    # Frozen features and development are shared. Live jobs receive isolated copies.
    encoders={};streams={};features={};controllers={};performed=0;completed=0;failures=0
    status='completed';last_notice=time.perf_counter();last_archive=time.perf_counter()
    try:
        for position,job in enumerate(jobs):
            path=out/'jobs'/(job['id']+'.json')
            if path.exists():
                saved=json.loads(path.read_text())
                if saved.get('status')=='completed' and not saved.get('failures'):
                    completed+=1;continue
            if cfg.max_jobs is not None and performed>=cfg.max_jobs:
                status='stopped_at_job_limit';break
            key=(job['seed'],job['encoder'])
            if key not in encoders:
                print('Developing %s, seed %d.'%(job['encoder'],job['seed']),flush=True)
                encoders[key]=v24_develop_encoder(cfg,job['seed'],job['encoder'])
            base,development=encoders[key]
            variant=job.get('encoder_variant') or {}
            encoder=base.clone().configure(variant) if variant else base
            stream_key=(job['seed'],job['task'],job['order'])
            if stream_key not in streams:
                source=(v24_structured_stream(cfg,job['seed']) if job['task'].startswith('structured')
                        else build_stream(cfg,job['seed']))
                streams[stream_key]=v24_perturb_stream(source,cfg,job['seed'],job['task'],job['order'])
            stream=streams[stream_key]
            if job['seed'] not in controllers:
                codes=make_codebook(job['seed']);controllers[job['seed']]=(codes,Route(anatomy,job['seed'],codes))
            codes,route=controllers[job['seed']]
            holder=[]
            def create_model():
                model=v24_make_memory(job['memory'],job['seed'],route,codes,cfg)
                holder.append(model);return model
            started=time.perf_counter()
            try:
                if job.get('teaching'):
                    result=v24_run_teaching_job(cfg,job,encoder.clone(),stream,create_model)
                    actual=v24_array_costs(holder[-1])
                    trace_bytes=result['costs'].get('eligibility_mutable_bytes',0)
                    actual['mutable_numeric_bytes']+=trace_bytes
                    actual['total_memory_numeric_bytes']+=trace_bytes
                    result['costs'].update(actual)
                    result['costs']['external_presentations_completed']=holder[-1].t
                    result['costs']['final_state_sha256']=holder[-1].state_digest()
                    result['costs']['total_learning_presentations']=holder[-1].t+result['costs'].get('extra_labeled_presentations',0)
                    for row in result.get('metrics',[]):
                        row.setdefault('external_presentations',row.get('external'))
                        row.setdefault('extra_labeled_presentations',0)
                else:
                    cache=None
                    if not variant:
                        feature_key=key+(job['task'],job['memory'].get('feature','sparse'))
                        if feature_key not in features:
                            features[feature_key]=v24_cached_features(encoder,stream,job['memory'],cfg)
                        cache=features[feature_key]
                    result=v24_run_standard_job(cfg,job,encoder,stream,create_model(),cache)
                result['status']='failed' if result.get('failures') else 'completed'
            except Exception as exc:
                result=dict(status='failed',failures=[dict(exception=type(exc).__name__,message=str(exc),
                    traceback=traceback.format_exc(limit=12))],metrics=[],diagnostics=[],
                    costs=dict(external_presentations_completed=holder[-1].t if holder else 0))
            result['job']=job;result['development']=development
            result['stream']={k:v for k,v in stream.items() if k in ('unique_identities','unique_inputs',
                'external','full_stream_sha256','prefix_check','prefix_costs','task_metadata','perturbation','ordering')}
            result.setdefault('timing',{})['total_job_seconds']=time.perf_counter()-started
            v24_atomic_json(path,result);performed+=1
            if result['status']=='completed':completed+=1
            else:
                failures+=1
                print('FAILED '+job['id']+': '+str(result['failures'][0].get('message')),flush=True)
            if performed%10==0 or time.perf_counter()-last_notice>25:
                print('%d/%d complete, %d failed; %s %s seed %d %s'%(
                    completed,len(jobs),failures,job['panel'],job['encoder'],job['seed'],job['memory']['name']),flush=True)
                last_notice=time.perf_counter()
            # Compact tables rebuilt periodically; every job already has an atomic checkpoint.
            if time.perf_counter()-last_archive>600:
                v24_write_outputs(out,cfg,jobs,verification,'running');last_archive=time.perf_counter()
    except KeyboardInterrupt:
        status='interrupted';print('\nInterrupted; completed individual jobs are resumable.',flush=True)
    except Exception as exc:
        status='runner_failed'
        v24_atomic_json(out/'RUNNER_ERROR.json',dict(exception=type(exc).__name__,message=str(exc),
            traceback=traceback.format_exc(limit=12)))
        print('Runner error: '+str(exc),flush=True)
    if failures and status=='completed':status='completed_with_failures'
    archive=v24_write_outputs(out,cfg,jobs,verification,status)
    print('Results ZIP: '+str(archive),flush=True)
    print('Status: %s; %d/%d completed.'%(status,completed,len(jobs)),flush=True)
    return 0 if status=='completed' else 2




# === V25 encoder pathways and minimal temporal challenger ===
# This fragment is assembled after the self-contained V24 reference source.

class V25LocalGateEncoder(V24Encoder):
    """Matched local novelty state on top of the existing isolated E3 circuit.

    Sensory Oja proposal magnitude is a novelty proxy, not demonstrated
    prediction inadequacy. Per-cell traces use EMA1/64; the first64 local bins
    are ungated warmup, then proposals above own previous mean+0.5SD pass.
    Both plastic afferents and intrinsic threshold changes receive this gate.
    """

    def __init__(self, base, enabled):
        self.__dict__ = _v24_copy.deepcopy(base.__dict__)
        self.novelty_enabled = bool(enabled)
        self.novelty_mean = np.zeros(self.N)
        self.novelty_second = np.zeros(self.N)
        self.novelty_seen = np.zeros(self.N,dtype=np.int64)
        self.novelty_considered = 0
        self.novelty_open = 0
        self.novelty_applied = 0

    def _forward_numpy(self, x, adapt=False, learning_rate=None):
        a = self.arrays; v = self.variant
        dtype = a['wm'].dtype
        x = np.asarray(x,dtype=dtype)
        B,T,D = x.shape
        if (T,D)!=(12,32):
            raise ValueError('Encoder requires twelve bins of 32 observed values.')
        if adapt and B != 1:
            raise ValueError('Causal encoder adaptation consumes one experience at a time.')
        if v['reset']=='continuous':
            m = np.repeat(self.private_state[None],B,axis=0)
            hist = np.repeat(self.private_history[None],B,axis=0)
        else:
            m = np.zeros((B,self.N,6),dtype=dtype)
            hist = np.zeros((B,3,self.N),dtype=dtype)
        tau = np.exp(math.log(1.5)+math.log(32./1.5)/(1.+np.exp(-a['tau_raw'])))
        alpha = -np.expm1(-1./tau)
        if v['timescales']=='single':
            # Match mean state-update rate, not arithmetic mean time constant.
            alpha = np.full_like(alpha,float(alpha.mean()))
        gain = np.logaddexp(0.,a['gain'])
        w1,w2 = a['w1'][a['type_index']], a['w2'][a['type_index']]
        b1,b2 = a['b1'][a['type_index']], a['b2'][a['type_index']]
        syn = self._syn().copy() if adapt else self._syn()
        histidx = ((a['delays']-1)*self.N+a['recurrent']).reshape(-1)
        plast = np.ones(self.N,dtype=dtype)
        plast[:round(self.N*v['stable_fraction'])] = 0
        rate = float(v['online_rate'] if learning_rate is None else learning_rate)
        do_learn = adapt and rate > 0
        for t in range(T):
            if t == getattr(self,'reset_at_bin',None):
                # Controlled removal of carried context after observing the
                # full prefix: reset state/history, retain current sensory x.
                m.fill(0.)
                hist.fill(0.)
            sensed = x[:,t][:,a['sensory']]
            rec = hist.reshape(B,-1)[:,histidx].reshape(B,self.N,8)
            if v['recurrence']=='off': rec = np.zeros_like(rec)
            elif v['recurrence']=='pooled': rec = np.broadcast_to(rec.mean(-1,keepdims=True),rec.shape)
            elif v['recurrence']=='coincidence': rec = rec*np.tanh(np.abs(sensed).mean(-1))
            branch = np.tanh(gain*((sensed*syn[None,:,:,:3]).sum(-1)+rec*syn[None,:,:,3]+a['bias']))
            z = np.concatenate((branch,m),axis=-1)
            hidden = np.tanh(np.einsum('bnq,nqh->bnh',z,w1,optimize=False)+b1)
            candidate = np.tanh(np.einsum('bnh,nhm->bnm',hidden,w2,optimize=False)+b2)
            write = alpha[None]
            forget = alpha[None]
            if v['stubborn_fraction']:
                n = round(self.N*v['stubborn_fraction'])
                write = np.broadcast_to(write,m.shape).copy()
                write[:,:n] /= 1.+4.*np.abs(candidate[:,:n]-m[:,:n])
                forget = write
            if v['gates']=='coupled':
                gate = .5+.5*np.tanh(np.abs(branch).mean(-1,keepdims=True))
                write = write*gate; forget = write
            elif v['gates']=='independent':
                incoming = .5+.5*np.tanh(np.abs(branch).mean(-1,keepdims=True))
                retaining = .5+.5*np.tanh(np.abs(m))
                write = write*incoming; forget = forget*retaining
            m = np.clip((1.-forget)*m+write*candidate,-1.,1.)
            voltage = (m*a['wm']).sum(-1)+(branch*a['wd']).sum(-1)+a['abias']
            if v['competition']=='inhibitory' or v['inhibitory_fixed'] or v['inhibitory_learning']:
                voltage = voltage - hist[:,0,self.inhibitory_sources]*self.inhibitory_weights
            threshold = np.logaddexp(0.,a['threshold'])
            activity = 1./(1.+np.exp(np.clip(-4.*(voltage-threshold),-60,60)))
            if do_learn:
                # Inhibition broadcasts only present population activity.
                gate = np.maximum(activity[0]-activity[0].mean(),0.)
                gate = np.minimum(8.*gate,1.)*plast
                post = branch[0]
                pre = sensed[0]
                # Each receiving cell measures only its own Oja proposal. This
                # is a sensory novelty proxy, never a label or task-change flag.
                proposal = post[:,:,None]*pre-post[:,:,None]**2*syn[:,:,:3]
                magnitude = np.sqrt(np.mean(proposal**2,axis=(1,2)))
                variance = np.maximum(self.novelty_second-self.novelty_mean**2,0.)
                threshold = self.novelty_mean+.5*np.sqrt(variance+1e-12)
                candidate_gate = ((self.novelty_seen < 64) | (magnitude > threshold)).astype(dtype)
                novelty = candidate_gate if self.novelty_enabled else np.ones(self.N,dtype=dtype)
                self.novelty_considered += int(np.count_nonzero(plast))
                self.novelty_open += int(np.count_nonzero(candidate_gate*plast))
                self.novelty_applied += int(np.count_nonzero(novelty*plast))
                # The no-gate matched control stores and updates these same
                # local traces; only multiplication into the update differs.
                trace_rate = 1./64.
                self.novelty_mean += trace_rate*plast*(magnitude-self.novelty_mean)
                self.novelty_second += trace_rate*plast*(magnitude**2-self.novelty_second)
                self.novelty_seen += (plast>0).astype(np.int64)
                gate *= novelty
                syn[:,:,:3] += rate*gate[:,None,None]*proposal
                if v['recurrent_learning']:
                    syn[:,:,3] += rate*gate[:,None]*(post*rec[0]-post**2*syn[:,:,3])
                np.clip(syn,-2,2,out=syn)
                a['threshold'] += rate*.25*plast*(activity[0]-.15)*novelty
                np.clip(a['threshold'],-5,3,out=a['threshold'])
                self.activity_rate += .01*(activity[0]-self.activity_rate)
                if v['inhibitory_learning']:
                    # Increased coincident excess activity strengthens its
                    # receiving-cell inhibitory contact; no covariance solver.
                    self.inhibitory_weights += rate*plast*hist[0,0,self.inhibitory_sources]*(activity[0]-.15)
                    np.clip(self.inhibitory_weights,0,1,out=self.inhibitory_weights)
                if self.candidate_sources is not None:
                    cpre = x[0,t,self.candidate_sources]
                    self.candidate_scores *= .995
                    self.candidate_scores += .005*np.abs(post[:,:,None]*cpre)
                    self.contact_scores *= .995
                    self.contact_scores += .005*np.abs(post[:,:,None]*pre)
                self.local_steps += 1
            hist = np.concatenate((activity[:,None],hist[:,:2]),axis=1)
        if adapt:
            if do_learn: self._write_syn(syn)
            self.sensory_presentations += 1
            if v['reset']=='continuous':
                self.private_state[...] = m[0]
                self.private_history[...] = hist[0]
            if do_learn and v['turnover']=='local' and self.sensory_presentations % 48 == 0:
                # One local branch contact per cell competes with its bounded
                # candidate contacts. Retained control pays identical traces.
                for i in range(self.N):
                    if not plast[i]: continue
                    r = (self.sensory_presentations//48+i) % 8
                    old = int(np.argmin(self.contact_scores[i,r]))
                    new = int(np.argmax(self.candidate_scores[i,r]))
                    source = int(self.candidate_sources[i,r,new])
                    if (source not in a['sensory'][i,r] and
                            self.candidate_scores[i,r,new] > self.contact_scores[i,r,old]):
                        a['sensory'][i,r,old] = source
                        syn[i,r,old] = .1
                        self.contact_scores[i,r,old] = self.candidate_scores[i,r,new]
                        self.rewired_contacts += 1
                self._write_syn(syn)
        return activity

    def state_digest(self):
        h = hashlib.sha256(super().state_digest().encode())
        h.update(str((self.novelty_enabled,self.novelty_considered,
            self.novelty_open,self.novelty_applied)).encode())
        for a in (self.novelty_mean,self.novelty_second,self.novelty_seen):
            h.update(a.tobytes())
        return h.hexdigest()


class V25ContextResetEncoder(V24Encoder):
    """Frozen E3 with a literal state reset just before the late cue at bin5.

    The full early context and distractor are processed. Only the six private
    states and three activity-history bins are cleared at the declared event
    boundary. Afferents, microdynamics, thresholds and late sensory input are
    identical to the frozen reference; this class adds no numeric arrays.
    """

    def __init__(self,base):
        self.__dict__ = _v24_copy.deepcopy(base.__dict__)
        self.variant['online_rate'] = 0.
        self.reset_at_bin = 5

    def _forward_numpy(self,x,adapt=False,learning_rate=None):
        if adapt or (learning_rate is not None and learning_rate != 0):
            raise ValueError('The context-reset diagnostic has a frozen encoder.')
        # Reuse the copied temporal arithmetic. The novelty branch cannot run
        # because no adaptation or positive sensory learning rate is allowed.
        return V25LocalGateEncoder._forward_numpy(self,x,False,0.)

    def state_digest(self):
        h = hashlib.sha256(super().state_digest().encode())
        h.update(str(('reset_private_state_and_history_before_bin',self.reset_at_bin)).encode())
        return h.hexdigest()


def v25_encoder_array_inventory(enc):
    """Enumerate actual persistent numeric arrays, including prescribed ones."""
    arrays = dict(enc.arrays)
    if getattr(enc,'model',None) is not None:
        arrays.update({name:a.detach().cpu().numpy()
            for name,a in enc.model.state_dict().items()})
    for name in ('mean','sd','private_state','private_history',
                 'inhibitory_sources','inhibitory_weights','activity_rate',
                 'synapse_q8','candidate_sources','candidate_scores',
                 'contact_scores','novelty_mean','novelty_second','novelty_seen'):
        a = getattr(enc,name,None)
        if a is not None:
            arrays[name] = a
    return arrays


class V25MinimalEncoder:
    """Fixed eight-contact expansion plus one leaky temporal trace per cell.

    Tau repeats (2,6,16) across cells, providing three distinct temporal
    weightings. There is no recurrence, gradient training, shared micro-MLP,
    or persistent cross-episode state. Exact top-k and Euclidean population
    normalization are engineering operations, stated explicitly.
    """

    def __init__(self,cfg,seed):
        self.N = int(cfg.neurons)
        if self.N != 96:
            raise ValueError('V25 fixes content width at96.')
        self.active_fraction = float(cfg.active_fraction)
        rg = rng_for(seed,'v25_minimal_sparse_afferents')
        sensory = np.stack([rg.choice(32,8,replace=False) for _ in range(self.N)])
        weights = rg.normal(0,1/math.sqrt(8),(self.N,8))
        tau = np.asarray([2.,6.,16.])[np.arange(self.N)%3]
        self.arrays = dict(sensory=sensory.astype(np.int64),weights=weights,
                          alpha=-np.expm1(-1./tau))
        self.mean = np.zeros(self.N)
        self.sd = np.ones(self.N)
        self.local_steps = 0
        self.sensory_presentations = 0
        self.regime = 'minimal_fixed_sparse_temporal'
        self.variant = dict(online_rate=0.,isolate_fraction=.5,
                            active_fraction=self.active_fraction,reset='experience')
        rg = rng_for(seed,'development_data')
        bases = cue_bases(rg,cfg.dev_cues)
        x = render_cues(bases,1,rg,cfg.noise)[:,0]
        calibration = self.activities(x)
        self.mean = calibration.mean(0)
        self.sd = np.maximum(calibration.std(0),.01)
        self.development = dict(
            encoder_regime=self.regime,
            development_unique_inputs=len(x),
            development_training_presentations=0,
            development_calibration_presentations=len(x),
            development_independent_noisy_views_per_identity=1,
            outcome_labels_used=0,hidden_backpropagation=False,
            online_synaptic_backpropagation=False,
            development_target='none; prescribed sparse afferents',
            calibration_policy='same separately rendered V24 development identities; frozen mean/sd',
            local_rule='none; fixed afferents',
            temporal_rule='one trace per cell, tau2/6/16, reset at each12-bin episode',
            population_operations='exact absolute top-k and normalization separately within48-cell halves')

    def activities(self,x,adapt=False):
        x = np.asarray(x,dtype=np.float64)
        if x.ndim == 2:
            x = x[None]
        if x.ndim != 3 or x.shape[-2:] != (12,32):
            raise ValueError('Expected (batch,12,32) observed sensory inputs.')
        m = np.zeros((len(x),self.N))
        for t in range(12):
            drive = np.tanh((x[:,t][:,self.arrays['sensory']]*self.arrays['weights']).sum(-1))
            m += self.arrays['alpha']*(drive-m)
        if adapt:
            self.sensory_presentations += len(x)
        return m

    def features(self,x,adapt=False):
        z = np.clip((self.activities(x,adapt=adapt)-self.mean)/self.sd,-5.,5.)
        parts = []
        for left,right in ((0,48),(48,96)):
            part = z[:,left:right].copy()
            k = max(1,round(48*self.active_fraction))
            chosen = np.argpartition(np.abs(part),-k,axis=1)[:,-k:]
            h = np.zeros_like(part)
            np.put_along_axis(h,chosen,np.take_along_axis(part,chosen,axis=1),axis=1)
            h /= np.maximum(np.linalg.norm(h,axis=1,keepdims=True),1e-12)
            parts.append(h/math.sqrt(2))
        return np.concatenate(parts,axis=1)

    def state_digest(self):
        h = hashlib.sha256(json.dumps(self.variant,sort_keys=True).encode())
        for name,a in sorted(v25_encoder_array_inventory(self).items()):
            h.update(name.encode());h.update(a.tobytes())
        h.update(str(self.sensory_presentations).encode())
        return h.hexdigest()


class V25ObservableEncoder:
    """Diagnostic with explicit knowledge of the two observable pulse bins.

    Copies observed bins0 and5 in order; no latent clean patterns or labels.
    Uses64 coordinates padded with32 zeros; no bank-independent learned map.
    This task-informed baseline is intentionally neither sparse nor ELM-like.
    """

    def __init__(self,cfg,seed):
        self.N = 96
        self.arrays = {}
        self.local_steps = 0
        self.sensory_presentations = 0
        self.variant = dict(online_rate=0.,reset='experience',pulse_bins=[0,5])
        self.development = dict(encoder_regime='observable_ordered_pulses',
            development_unique_inputs=0,development_training_presentations=0,
            development_calibration_presentations=0,outcome_labels_used=0,
            hidden_backpropagation=False,online_synaptic_backpropagation=False,
            task_prior='known observed pulse positions0 and5; ordered concatenation',
            local_rule='none; observable-input diagnostic',
            population_operations='global Euclidean normalization; dense64 coordinates plus32 zero padding')

    def features(self,x,adapt=False):
        x = np.asarray(x,dtype=np.float64)
        if x.ndim == 2:
            x = x[None]
        if x.ndim != 3 or x.shape[-2:] != (12,32):
            raise ValueError('Expected (batch,12,32) observed sensory inputs.')
        z = np.zeros((len(x),96),dtype=np.float64)
        z[:,:64] = np.concatenate((x[:,0],x[:,5]),axis=-1)
        z /= np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-12)
        if adapt:
            self.sensory_presentations += len(x)
        return z

    def state_digest(self):
        return hashlib.sha256(json.dumps(dict(self.variant,
            sensory_presentations=self.sensory_presentations),sort_keys=True).encode()).hexdigest()


class V25EncoderBundle:
    """Frozen/live address-value branches with a shared causal sensory clock.

    For live E3, one adapt=True call preserves V24's exact temporal order:
    local updates occur inside the12-bin experience, and the returned activity
    is that trajectory's final activity. It is not recomputed after learning.
    No outcome, cue ID, historical destination, or evaluator state is accepted.
    adapt=False probes never change model state. Work counters distinguish
    immutable evaluation bookkeeping from learned/circuit state.
    """

    KINDS = ('dual','shared_frozen','shared_live','shared_context_reset','split_frozen','split_live',
             'split_stable','split_stable_gated','minimal','observable')

    def __init__(self,cfg,seed,kind,base=None):
        if kind not in self.KINDS:
            raise ValueError('Unknown V25 encoder kind: '+str(kind))
        if int(cfg.neurons) != 96:
            raise ValueError('V25 fixes encoder content width at96.')
        self.kind,self.seed = kind,int(seed)
        self.N = 96
        self.online_rate = .0002
        self.encoders = {}
        if kind == 'minimal':
            self.encoders['S'] = V25MinimalEncoder(cfg,seed)
        elif kind == 'observable':
            self.encoders['S'] = V25ObservableEncoder(cfg,seed)
        else:
            if base is None:
                base,_ = v24_develop_encoder(cfg,seed,'E3')
            if base.regime != 'E3' or base.model is not None:
                raise ValueError('V25 candidates must start from locally developed E3.')
            if kind == 'dual':
                self.encoders['F'] = base.clone().configure(dict(online_rate=0.))
                self.encoders['L'] = base.clone().configure(dict(online_rate=self.online_rate))
            else:
                enc = base.clone()
                if kind.startswith('split'):
                    enc.configure(dict(isolate_fraction=.5,
                        stable_fraction=.5 if kind.startswith('split_stable') else 0.,
                        online_rate=0. if kind=='split_frozen' else self.online_rate))
                    if kind in ('split_stable','split_stable_gated'):
                        enc = V25LocalGateEncoder(enc,enabled=kind.endswith('_gated'))
                else:
                    enc.configure(dict(online_rate=self.online_rate if kind=='shared_live' else 0.))
                    if kind == 'shared_context_reset':
                        enc = V25ContextResetEncoder(enc)
                self.encoders['S'] = enc
        self.training_presentations = 0
        self.training_branch_evaluations = 0
        self.probe_branch_evaluations = 0
        self.feature_calls = 0
        self._initial_local_steps = {k:int(getattr(e,'local_steps',0)) for k,e in self.encoders.items()}
        self.development = _v24_copy.deepcopy(next(iter(self.encoders.values())).development)
        self.development.update(
            encoder_bundle_kind=kind,
            development_runs_per_simulated_learner=0 if kind=='observable' else 1,
            frozen_and_live_branches_share_one_initial_development=(kind=='dual'),
            persistent_encoder_copies=len(self.encoders),
            local_online_rate=self.online_rate if any(e.variant['online_rate']>0 for e in self.encoders.values()) else 0.,
            training_encoding_order='V24 causal within-episode sensory update; final trajectory activity; no second encode',
            dual_branch_policy='both frozen and live branches process every external experience' if kind=='dual' else 'one branch',
            online_calibration_policy='development mean/sd frozen; no probe statistics used',
            within_episode_context_reset='clear six private states and three activity histories before bin5; full sensory prefix processed; late cue preserved' if kind=='shared_context_reset' else 'none',
            novelty_gate='receiving-cell sensory Oja proposal RMS > prior EMA mean+.5SD; EMA1/64, warmup64bins' if kind.endswith('_gated') else 'off',
            novelty_matched_trace_state=kind in ('split_stable','split_stable_gated'))

    def clone(self):
        return _v24_copy.deepcopy(self)

    copy = clone

    @property
    def arrays(self):
        return {branch+'.'+name:a for branch,enc in self.encoders.items()
            for name,a in v25_encoder_array_inventory(enc).items()}

    def features(self,x,adapt=False):
        x = np.asarray(x)
        if x.ndim == 2:
            x = x[None]
        if x.ndim != 3 or x.shape[-2:] != (12,32):
            raise ValueError('Expected sensory array of shape(batch,12,32).')
        if not np.isfinite(x).all():
            raise ValueError('Sensory inputs must be finite.')
        # Batch adaptation is processed in external order. This also makes
        # batching an execution choice that cannot alter local learning.
        if adapt and len(x)>1:
            rows = [self.features(row,adapt=True) for row in x]
            return {k:np.concatenate([r[k] for r in rows],axis=0) for k in self.encoders}
        result = {}
        for branch,enc in self.encoders.items():
            learn = bool(adapt and enc.variant['online_rate']>0)
            result[branch] = np.asarray(enc.features(x,adapt=learn),dtype=np.float64)
            if result[branch].shape != (len(x),96) or not np.isfinite(result[branch]).all():
                raise FloatingPointError('Invalid encoder feature result.')
        if adapt:
            self.training_presentations += len(x)
            self.training_branch_evaluations += len(x)*len(self.encoders)
        else:
            self.probe_branch_evaluations += len(x)*len(self.encoders)
        self.feature_calls += 1
        return result

    def state_digest(self):
        h = hashlib.sha256(str((self.kind,self.seed,self.training_presentations)).encode())
        for branch,enc in sorted(self.encoders.items()):
            h.update(branch.encode());h.update(enc.state_digest().encode())
        # Pure probe calls update only execution counters; they are expressly
        # outside circuit/learning state, and omitted from this digest.
        return h.hexdigest()

    def costs(self):
        inventory = []
        mutable_names = {'syn','synapse_q8','threshold','activity_rate',
            'novelty_mean','novelty_second','novelty_seen'}
        for branch,enc in self.encoders.items():
            for name,a in v25_encoder_array_inventory(enc).items():
                live = enc.variant['online_rate']>0 and name in mutable_names
                inventory.append(dict(branch=branch,name=name,shape=list(a.shape),
                    dtype=str(a.dtype),bytes=int(a.nbytes),online_mutable=bool(live)))
        actual = sum(a['bytes'] for a in inventory)
        mutable = sum(a['bytes'] for a in inventory if a['online_mutable'])
        local_steps = sum(int(getattr(e,'local_steps',0))-self._initial_local_steps[k]
            for k,e in self.encoders.items())
        result = dict(encoder_array_bytes=int(actual),encoder_fixed_array_bytes=int(actual-mutable),
            encoder_mutable_array_bytes=int(mutable),encoder_numeric_inventory=inventory,
            encoder_branch_count=len(self.encoders),encoder_content_width=96,
            encoder_local_time_bins=int(local_steps),
            encoder_sensory_presentations=self.training_presentations,
            encoder_training_branch_evaluations=self.training_branch_evaluations,
            encoder_probe_branch_evaluations=self.probe_branch_evaluations,
            encoder_total_branch_evaluations=self.training_branch_evaluations+self.probe_branch_evaluations,
            encoder_training_time_bin_evaluations=12*self.training_branch_evaluations,
            encoder_probe_time_bin_evaluations=12*self.probe_branch_evaluations,
            encoder_feature_calls=self.feature_calls,
            encoder_array_accounting_excludes='Python objects, scalar metadata and temporary forward/evaluator arrays',
            encoder_fixed_arrays_charged=True,
            encoder_mutability_accounting='whole stored arrays; stable half portions remain counted in allocated mutable arrays')
        if self.kind == 'observable':
            result.update(encoder_training_time_bin_evaluations=0,
                encoder_probe_time_bin_evaluations=0,
                observable_copied_values_per_experience=64)
        gated = [e for e in self.encoders.values() if isinstance(e,V25LocalGateEncoder)]
        if gated:
            considered = sum(e.novelty_considered for e in gated)
            result.update(encoder_novelty_considered_cell_bins=considered,
                encoder_novelty_open_fraction=sum(e.novelty_open for e in gated)/max(1,considered),
                encoder_novelty_applied_fraction=sum(e.novelty_applied for e in gated)/max(1,considered),
                encoder_novelty_trace_bytes=int(sum(e.novelty_mean.nbytes+e.novelty_second.nbytes+e.novelty_seen.nbytes for e in gated)))
        return result


def v25_encoder_checks(cfg,base=None):
    """Small semantic checks. These do not train any experimental trajectory."""
    seed = 17
    if base is None:
        base,_ = v24_develop_encoder(cfg,seed,'E3')
    else:
        seed = int(base.seed)
    rg = rng_for(seed,'v25_encoder_interface_checks')
    x = render_cues(cue_bases(rg,5),1,rg,.12)[:,0]
    dual = V25EncoderBundle(cfg,seed,'dual',base)
    initial = dual.features(x)
    assert np.array_equal(initial['F'],initial['L']), 'Dual branches must initialize identically.'
    digest = dual.state_digest();dual.features(x)
    assert digest == dual.state_digest(), 'Probe encoding changed learned state.'
    single = V25EncoderBundle(cfg,seed,'shared_live',base)
    reference = base.clone().configure(dict(online_rate=.0002))
    for row in x[:2]:
        expected = reference.features(row,adapt=True)
        got = single.features(row,adapt=True)['S']
        assert np.array_equal(expected,got), 'E3 live training diverged from V24 arithmetic.'
        pair = dual.features(row,adapt=True)
        assert np.array_equal(pair['L'],got), 'Dual live differs from the shared live branch.'
    assert np.array_equal(initial['F'],dual.features(x)['F']), 'Frozen dual branch drifted.'
    assert not np.array_equal(initial['L'],dual.features(x)['L']), 'Live dual branch did not adapt.'
    assert dual.costs()['encoder_array_bytes'] == 2*single.costs()['encoder_array_bytes']
    assert dual.training_branch_evaluations == 2*single.training_branch_evaluations
    frozen = V25EncoderBundle(cfg,seed,'shared_frozen',base)
    assert np.array_equal(frozen.features(x)['S'],base.features(x)), 'Frozen E3 continuity failed.'
    stable = V25EncoderBundle(cfg,seed,'split_stable',base)
    gated = V25EncoderBundle(cfg,seed,'split_stable_gated',base)
    expected_stable = base.clone().configure(dict(isolate_fraction=.5,stable_fraction=.5,online_rate=.0002))
    before = stable.features(x)['S']
    for row in x:
        got = stable.features(row,adapt=True)['S']
        expected = expected_stable.features(row,adapt=True)
        assert np.array_equal(got,expected), 'Matched no-gate split altered V24 arithmetic.'
        gated.features(row,adapt=True)
    assert np.array_equal(before[:,:48],stable.features(x)['S'][:,:48]), 'Stable address coordinates drifted.'
    for _ in range(3):
        for row in x:
            gated.features(row,adapt=True)
    assert np.array_equal(before[:,:48],gated.features(x)['S'][:,:48]), 'Gated stable address coordinates drifted.'
    assert stable.costs()['encoder_array_bytes'] == gated.costs()['encoder_array_bytes']
    assert gated.costs()['encoder_novelty_applied_fraction'] < 1., 'Novelty gate never suppressed an update.'
    assert stable.costs()['encoder_novelty_applied_fraction'] == 1., 'No-gate matched control suppressed updates.'
    minimal = V25EncoderBundle(cfg,seed,'minimal')
    ordered = minimal.features(x)['S']
    swap = x.copy();swap[:,:3] = x[:,5:8];swap[:,5:8] = x[:,:3]
    order_effect = float(np.mean((ordered-minimal.features(swap)['S'])**2))
    assert order_effect > 1e-8, 'Minimal temporal encoder erased pulse order.'
    assert np.max((np.abs(ordered)>0).sum(-1)) == 24
    observable = V25EncoderBundle(cfg,seed,'observable')
    z = observable.features(x)['S']
    raw = np.concatenate((x[:,0],x[:,5]),axis=-1).astype(np.float64)
    raw /= np.maximum(np.linalg.norm(raw,axis=1,keepdims=True),1e-12)
    assert np.array_equal(z[:,:64],raw) and not np.any(z[:,64:])
    reset = V25EncoderBundle(cfg,seed,'shared_context_reset',base)
    changed_prefix = x.copy()
    changed_prefix[:,:5] = -changed_prefix[:,:5]
    reset_original = reset.features(x)['S']
    reset_changed = reset.features(changed_prefix)['S']
    assert np.array_equal(reset_original,reset_changed), 'Literal state reset retained early context.'
    reference_context_effect = float(np.mean((frozen.features(x)['S']-
        frozen.features(changed_prefix)['S'])**2))
    assert reference_context_effect > 1e-12, 'Context reset check did not manipulate a represented prefix.'
    assert reset.costs()['encoder_array_bytes'] == frozen.costs()['encoder_array_bytes'], 'Context reset added unmatched array state.'
    for kind in V25EncoderBundle.KINDS:
        bundle = V25EncoderBundle(cfg,seed,kind,base)
        digest = bundle.state_digest()
        result = bundle.features(x)
        assert digest == bundle.state_digest(), (kind,'Probe mutation')
        assert all(v.shape==(len(x),96) and np.isfinite(v).all() for v in result.values())
        assert bundle.costs()['encoder_array_bytes'] == sum(a.nbytes for a in bundle.arrays.values())
    return dict(dual_identical_initialization=True,live_E3_exact_V24_arithmetic=True,
        frozen_E3_exact_V24_arithmetic=True,dual_state_compute_charged=True,
        pure_probe_reads=True,stable_route_coordinates_exactly_preserved=True,
        matched_no_gate_exact_V24_arithmetic=True,novelty_gate_active=True,
        minimal_pulse_order_score_displacement=order_effect,
        observable_uses_only_ordered_observed_values=True,
        context_reset_clears_private_states_and_activity_history=True,
        context_reset_has_matched_array_bytes=True,
        context_reference_prefix_score_displacement=reference_context_effect,
        training_encoding_order='within-experience local updates, one final activity code as V24')



# V25: separate address and value pathways with local bank learning.
class V25BankMemory(MemoryBase):
    """Bank-local delta memory. Routing sees only r; stored values see only z.

    `r == z`, full-width float64, and ordinary spill settings reproduce V24.
    Read views are pure and never alter the selected-bank teaching error.
    Int8 has an actual packed coefficient array: no persistent float shadow.
    Public V is a temporary dequantized view for int8; use its setter to write.
    """
    def __init__(self, d, seed, address_dim=None, banks=8, routing='fixed',
                 eta=.6, protection='none', release=False, precision='float64',
                 spillover=0., normalize_spill=False, strength=3., recovery=.997,
                 address_eta=.05, temperature=8.):
        super().__init__(d, seed)
        self.address_dim = self.d if address_dim is None else int(address_dim)
        self.banks = int(banks)
        if self.d < 1 or self.address_dim < 1 or self.banks < 1:
            raise ValueError('Positive value width, address width, and bank count required')
        self.routing = {'learned': 'local', 'adaptive': 'local'}.get(routing, routing)
        if self.routing not in ('fixed', 'local'):
            raise ValueError('Routing is fixed or local')
        self.eta = float(eta)
        self.protection = str(protection)
        self.release = bool(release)
        self.precision = {'int8': 'quantized8', 'float': 'float64'}.get(precision, precision)
        self.spillover = float(spillover)
        self.normalize_spill = bool(normalize_spill)
        self.strength = float(strength)
        self.recovery = float(recovery)
        self.address_eta = float(address_eta)
        self.temperature = float(temperature)
        if self.protection not in ('none', 'neuron', 'pair'):
            raise ValueError('Protection is none, neuron, or pair')
        if self.normalize_spill and self.spillover and self.protection != 'none':
            raise ValueError('Norm-matched spill is defined only for unprotected writes')
        if self.precision not in ('float64', 'float32', 'quantized8'):
            raise ValueError('Precision is float64, float32, or int8/quantized8')
        if not (0 <= self.spillover <= 1 and 0 <= self.recovery <= 1):
            raise ValueError('Spillover and recovery must be in [0,1]')
        if not (self.eta >= 0 and self.strength >= 0 and 0 <= self.address_eta <= 1 and self.temperature > 0):
            raise ValueError('Invalid local plasticity or read-temperature parameter')
        if not all(math.isfinite(x) for x in (self.eta, self.spillover, self.strength,
                    self.recovery, self.address_eta, self.temperature)):
            raise ValueError('All numeric parameters must be finite')
        # Keep the historical RNG draw exactly for matched address dimensions.
        self.centers = unit_rows(self.rng.normal(size=(self.banks, self.address_dim)))
        self.bound = 4.  # Fixed V24 quantization range, not fit to evaluation data.
        dtype = {'float64': np.float64, 'float32': np.float32,
                 'quantized8': np.int8}[self.precision]
        self._weights = np.zeros((self.banks, self.d, 4), dtype=dtype)
        self.counts = np.zeros(self.banks, dtype=np.int64)
        self.last_seen = np.zeros(self.banks, dtype=np.int64)
        self.active = -1
        self.route_changes = 0
        self.releases = 0
        self.gain_sum = 0.
        self.correction_sum = 0.
        self.empty_write_events = 0
        self.feedback_events = 0
        self.feedback_bank_updates = 0
        self.separated_teaching = False
        self.quantization_clipped_values = 0
        self.importance = None
        self.surprise_ema = None
        if self.protection != 'none':
            channels = 1 if self.protection == 'neuron' else 4
            self.importance = np.zeros((self.banks, self.d, channels))
            self.surprise_ema = np.ones(self.banks)
        # Identity trace wiring needs no stored permutation array. This is the
        # same intact local protection computation, with its metadata removed.

    @property
    def V(self):
        if self.precision == 'quantized8':
            return self._weights.astype(np.float64) * (self.bound / 127.)
        return self._weights

    @V.setter
    def V(self, value):
        values = np.asarray(value)
        if values.shape != (self.banks, self.d, 4):
            raise ValueError('Wrong bank coefficient shape')
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite bank coefficient')
        if self.precision == 'quantized8':
            self.quantization_clipped_values += int(np.count_nonzero(np.abs(values) > self.bound))
            self._weights[...] = np.rint(np.clip(values / self.bound, -1., 1.) * 127).astype(np.int8)
        else:
            self._weights[...] = values

    def _v25_rows(self, x, width):
        x = np.asarray(x, dtype=np.float64)
        scalar = x.ndim == 1
        if scalar:
            x = x[None, :]
        if x.ndim != 2 or x.shape[1] != width:
            raise ValueError('Expected a vector or row matrix of width %d' % width)
        if not np.isfinite(x).all():
            raise ValueError('Nonfinite sensory representation')
        return x, scalar

    def probabilities(self, R, view='hard'):
        R, scalar = self._v25_rows(R, self.address_dim)
        score = unit_rows(R) @ self.centers.T
        if view in ('hard', 'winner'):
            p = np.zeros_like(score)
            p[np.arange(len(R)), np.argmax(score, axis=1)] = 1.
        elif view in ('soft', 'mixture', 'blend', 'blended'):
            p = _structural_softmax(self.temperature * score)
        else:
            raise ValueError('Read view must be hard or blended')
        return p[0] if scalar else p

    def routes(self, R):
        p = self.probabilities(R, 'hard')
        if p.ndim == 1:
            return int(np.argmax(p))
        return np.argmax(p, axis=1)

    def output_pair(self, R, Z, view='hard'):
        R, scalar = self._v25_rows(R, self.address_dim)
        Z, zscalar = self._v25_rows(Z, self.d)
        if len(R) != len(Z) or scalar != zscalar:
            raise ValueError('Address and value batches must match')
        values = self.V
        if self.banks == 1:
            # Match PlainMemory's direct multiplication for the ordinary arm.
            result = Z @ values[0]
        else:
            p = self.probabilities(R, view)
            local = np.stack([Z @ values[b] for b in range(self.banks)], axis=1)
            result = np.einsum('nb,nbc->nc', p, local, optimize=False)
        return result[0] if scalar else result

    def _advance_router(self, bank, q):
        self.counts[bank] += 1
        if self.routing == 'local':
            rate = min(self.address_eta, 1. / float(self.counts[bank]))
            self.centers[bank] = unit_rows((1. - rate) * self.centers[bank] + rate * q)
        self.route_changes += int(self.active >= 0 and bank != self.active)
        self.active = bank
        self.t += 1
        self.last_seen[bank] = self.t

    def _single_pair(self, r, z):
        R, scalar = self._v25_rows(r, self.address_dim)
        Z, zscalar = self._v25_rows(z, self.d)
        if not scalar or not zscalar:
            raise ValueError('A learning event accepts one address and one value vector')
        return R[0], Z[0]

    def observe_pair(self, r, z, label):
        r, z = self._single_pair(r, z)
        if int(label) != label or not 0 <= int(label) < 4:
            raise ValueError('Categorical label must be in [0,3]')
        label = int(label)
        q = unit_rows(r)
        bank = int(np.argmax(self.centers @ q))
        target = target4(label)
        values = self.V
        old_values = values.copy() if self.precision != 'float64' else None
        prediction = z @ values[bank]
        error = target - prediction
        surprise = float(error @ error)
        active = np.abs(z) > 1e-12
        conflict = False
        if self.importance is not None:
            if self.release:
                self.importance *= self.recovery
                margin = float(np.max(np.delete(prediction, label)) - prediction[label])
                conflict = margin > .30 and surprise > 1.25 * self.surprise_ema[bank]
                if conflict:
                    self.importance[bank, active] *= .35
                    self.releases += 1
            gain = 1. / (1. + self.strength * self.importance[bank])
        else:
            gain = np.ones((self.d, 1))
        change = delta_change(z, error, self.eta) * gain
        # This focused control preserves the full proposed Frobenius norm of
        # the unprotected no-spill write. Other banks receive the same z/error.
        # Protected spill combinations are excluded to preserve that meaning.
        if self.normalize_spill and self.spillover and self.importance is not None:
            raise ValueError('Norm-matched spill is defined only for unprotected writes')
        norm_factor = (math.sqrt(1. + (self.banks - 1) * self.spillover ** 2)
                       if self.normalize_spill else 1.)
        change = change / norm_factor
        values[bank] += change
        norm2 = float(np.sum(change * change))
        if self.spillover:
            for other in range(self.banks):
                if other == bank:
                    continue
                extra = delta_change(z, error, self.eta * self.spillover) / norm_factor
                values[other] += extra
                norm2 += float(np.sum(extra * extra))
        achieved = float(np.linalg.norm(z @ change)) / max(float(np.linalg.norm(error)), 1e-12)
        if self.importance is not None:
            post_error = target - z @ values[bank]
            reliability = float(np.clip(1. - float(post_error @ post_error) / 2., .05, 1.))
            credit = (np.outer(z * z, .10 + target) if self.protection == 'pair'
                      else (.35 * z * z)[:, None])
            self.importance[bank] += reliability * credit
            np.minimum(self.importance[bank], 1e6, out=self.importance[bank])
            self.surprise_ema[bank] = .95 * self.surprise_ema[bank] + .05 * surprise
        mean_gain = float(gain[active].mean()) if np.any(active) else 1.
        self.gain_sum += mean_gain
        self.correction_sum += achieved
        self.empty_write_events += int(not np.any(active))
        self.feedback_events += 1
        self.feedback_bank_updates += 1 + ((self.banks - 1) if self.spillover else 0)
        self._advance_router(bank, q)
        if self.precision != 'float64':
            self.V = values
            realized = self.V - old_values
            update_norm = float(np.linalg.norm(realized))
            realized_correction = float(np.linalg.norm(z @ realized[bank])) / max(float(np.linalg.norm(error)), 1e-12)
        else:
            update_norm = float(norm2 ** .5)
            realized_correction = achieved
        return {'extra_labeled_presentations': 0, 'update_norm': update_norm,
                'proposed_update_norm': float(norm2 ** .5), 'selected_bank': bank,
                'release': int(conflict), 'achieved_correction': achieved,
                'realized_correction': realized_correction, 'plasticity_gain': mean_gain,
                'write_feature_energy': float(z @ z), 'spillover_strength': self.spillover,
                'spillover_norm_factor': norm_factor}

    def _teaching_guard(self):
        if self.protection != 'none' or self.spillover:
            raise ValueError('Separated teaching supports only unprotected no-spill banks')

    def begin_pair(self, r, z):
        """Sensory event: return selected destination/local prediction, then route adaptation.

        No target, phase, probe identity or historical destination is consulted.
        The external teacher may retain the returned feature and destination in
        its explicitly bounded eligibility trace; its cost is accounted there.
        """
        self._teaching_guard()
        self.separated_teaching = True
        r, z = self._single_pair(r, z)
        q = unit_rows(r)
        bank = int(np.argmax(self.centers @ q))
        prediction = (z @ self.V[bank]).copy()
        self._advance_router(bank, q)
        return bank, prediction

    def apply_feedback(self, destination, z, error, eta_scale=1., action=None):
        """Apply eligible local credit without a new sensory event or router update.

        For scalar reward an explicit action gates the output connection. The
        caller supplies already-computed local error, never an output target for
        unchosen actions. Multiple trace updates are separate feedback operations.
        """
        self._teaching_guard()
        bank = int(destination)
        if destination != bank or not 0 <= bank < self.banks:
            raise ValueError('Feedback destination outside bank range')
        Z, scalar = self._v25_rows(z, self.d)
        if not scalar:
            raise ValueError('Feedback takes one eligible value vector')
        z = Z[0]
        scale = float(eta_scale)
        if not math.isfinite(scale) or scale < 0:
            raise ValueError('Feedback scale must be finite and nonnegative')
        error = np.asarray(error, dtype=np.float64)
        if action is not None:
            if int(action) != action or not 0 <= int(action) < 4:
                raise ValueError('Action must be in [0,3]')
            selected_error = float(error) if error.ndim == 0 else float(error[int(action)])
            error = np.zeros(4)
            error[int(action)] = selected_error
        if error.shape != (4,) or not np.isfinite(error).all():
            raise ValueError('Feedback requires finite four-output error or an action-gated scalar')
        values = self.V
        previous = values[bank].copy()
        change = delta_change(z, error, self.eta * scale)
        values[bank] += change
        if self.precision != 'float64':
            self.V = values
        realized = self.V[bank] - previous
        self.feedback_events += 1
        self.feedback_bank_updates += 1
        self.empty_write_events += int(not np.any(np.abs(z) > 1e-12))
        self.gain_sum += 1.
        achieved = float(np.linalg.norm(z @ realized)) / max(float(np.linalg.norm(error)), 1e-12)
        self.correction_sum += achieved
        return {'extra_labeled_presentations': 0, 'selected_bank': bank,
                'update_norm': float(np.linalg.norm(realized)),
                'proposed_update_norm': float(np.linalg.norm(change)),
                'achieved_correction': achieved, 'release': 0,
                'write_feature_energy': float(z @ z), 'plasticity_gain': 1.}

    def mutable_arrays(self):
        arrays = {'weights': self._weights, 'counts': self.counts, 'last_seen': self.last_seen}
        if self.routing == 'local':
            arrays['centers'] = self.centers
        if self.importance is not None:
            arrays['importance'] = self.importance
            arrays['surprise_ema'] = self.surprise_ema
        return arrays

    def fixed_arrays(self):
        return {'centers': self.centers} if self.routing == 'fixed' else {}

    def state_digest(self):
        digest = hashlib.sha256()
        arrays = dict(self.mutable_arrays(), **self.fixed_arrays())
        for key, array in sorted(arrays.items()):
            digest.update(key.encode())
            digest.update(str(array.dtype).encode())
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
        scalars = {key: value for key, value in vars(self).items()
                   if isinstance(value, (str, int, float, bool, type(None)))}
        digest.update(json.dumps(scalars, sort_keys=True, allow_nan=False).encode())
        digest.update(json.dumps(self.rng.bit_generator.state, sort_keys=True).encode())
        return digest.hexdigest()

    def costs(self):
        mutable = self.mutable_arrays()
        fixed = self.fixed_arrays()
        inventory = {key: {'bytes': int(array.nbytes), 'dtype': str(array.dtype),
                          'shape': list(array.shape), 'mutable': key in mutable}
                     for key, array in dict(mutable, **fixed).items()}
        mutable_bytes = sum(array.nbytes for array in mutable.values())
        fixed_bytes = sum(array.nbytes for array in fixed.values())
        router = self.centers.nbytes + self.counts.nbytes + self.last_seen.nbytes
        protection = (0 if self.importance is None else
                      self.importance.nbytes + self.surprise_ema.nbytes)
        return {'persistent_bytes': int(mutable_bytes + fixed_bytes),
                'numeric_persistent_bytes': int(mutable_bytes + fixed_bytes),
                'mutable_bytes': int(mutable_bytes), 'fixed_bytes': int(fixed_bytes),
                'bank_weight_bytes': int(self._weights.nbytes), 'router_bytes': int(router),
                'protection_bytes': int(protection), 'array_inventory': inventory,
                'precision': self.precision, 'readout_coefficients': int(self._weights.size),
                'temporary_dequantized_weight_bytes': int(self._weights.size * 8)
                    if self.precision == 'quantized8' else 0,
                'feedback_events': int(self.feedback_events),
                'local_bank_updates': int(self.feedback_bank_updates),
                'cost_scope': 'Owned numeric arrays including fixed router; Python-object overhead excluded; evaluator and encoder state separate'}

    def diagnostics(self):
        total = int(self.counts.sum())
        fractions = self.counts.astype(float) / max(1, total)
        nonzero = fractions[fractions > 0]
        return {'bank_capacity': self.banks, 'used_banks': int(np.sum(self.counts > 0)),
                'route_changes_between_consecutive_inputs': self.route_changes,
                'bank_sensory_destination_counts': self.counts.tolist(),
                'bank_sensory_destination_fractions': fractions.tolist(),
                'bank_selected_write_counts': None if self.separated_teaching else self.counts.tolist(),
                'bank_selected_write_fractions': None if self.separated_teaching else fractions.tolist(),
                'bank_load_cv': float(np.std(self.counts) / max(float(np.mean(self.counts)), 1e-12)),
                'bank_load_entropy': float(-np.sum(nonzero * np.log(nonzero))),
                'readout_coefficients': int(self._weights.size),
                'accessible_features_per_bank': self.d, 'address_features': self.address_dim,
                'release_count': self.releases,
                'importance_mean': float(self.importance.mean()) if self.importance is not None else 0.,
                'mean_plasticity_gain': self.gain_sum / max(1, self.feedback_events),
                'mean_correction_fraction': self.correction_sum / max(1, self.feedback_events),
                'empty_eligible_write_events': self.empty_write_events,
                'external_sensory_events': self.t, 'feedback_events': self.feedback_events,
                'stored_weight_precision': self.precision,
                'stored_weight_bytes': int(self._weights.nbytes),
                'quantization_clipped_values': self.quantization_clipped_values,
                'normalized_spill': self.normalize_spill}


def v25_memory_checks():
    """Small deterministic mechanism checks; no experiment outcome tuning."""
    result = {}
    rng = np.random.default_rng(42017)
    H = unit_rows(rng.normal(size=(31, 12)))
    labels = rng.integers(0, 4, size=len(H))
    for routing in ('fixed', 'local'):
        for protection, release, spill in (('none', False, 0.), ('neuron', True, 0.),
                                            ('pair', True, 0.), ('none', False, .1)):
            old = V24BankMemory(12, 19, banks=4, routing=routing,
                    protection=protection, release=release, spillover=spill)
            new = V25BankMemory(12, 19, banks=4, routing=routing,
                    protection=protection, release=release, spillover=spill)
            for h, label in zip(H, labels):
                left = old.observe(h, int(label))
                right = new.observe_pair(h, h, int(label))
                assert np.array_equal(old.V, new.V), 'V24 weight equivalence'
                assert np.array_equal(old.centers, new.centers), 'V24 router equivalence'
                assert left['selected_bank'] == right['selected_bank']
                if old.importance is not None:
                    assert np.array_equal(old.importance, new.importance)
                for view in ('hard', 'soft'):
                    assert np.array_equal(old.output_view(H, view, 0.), new.output_pair(H, H, view))
            result['v24_exact_%s_%s_%s_%s' % (routing, protection, release, spill)] = True
    plain = PlainMemory(12, 23)
    single = V25BankMemory(12, 23, banks=1)
    for h, label in zip(H, labels):
        plain.observe(h, int(label))
        single.observe_pair(h, h, int(label))
        assert np.array_equal(plain.V, single.V[0])
    assert np.array_equal(plain.output(H), single.output_pair(H, H))
    result['ordinary_single_store_exact'] = True
    separated = V25BankMemory(12, 17, address_dim=5)
    R = unit_rows(rng.normal(size=(len(H), 5)))
    for r, h, label in zip(R, H, labels):
        separated.observe_pair(r, h, int(label))
    before = separated.state_digest()
    for view in ('hard', 'blended'):
        pred = separated.output_pair(R, H, view)
        assert pred.shape == (len(H), 4)
        assert np.isfinite(pred).all()
        separated.routes(R)
        separated.probabilities(R, view)
    assert before == separated.state_digest()
    result['separate_paths_pure_views'] = True
    for precision in ('float32', 'quantized8'):
        old = V24PrecisionMemory(V24BankMemory(12, 31, banks=4), precision)
        new = V25BankMemory(12, 31, banks=4, precision=precision)
        for h, label in zip(H, labels):
            old.observe(h, int(label))
            new.observe_pair(h, h, int(label))
            assert np.array_equal(old.packed, new._weights), 'Historical packed precision equivalence'
            assert np.array_equal(old.output_view(H), new.output_pair(H, H))
        arrays = [v for v in vars(new).values() if isinstance(v, np.ndarray)]
        assert sum(a.shape == (4, 12, 4) for a in arrays) == 1
        assert new.costs()['persistent_bytes'] == sum(a.nbytes for a in arrays)
        before = new.state_digest()
        new.output_pair(H, H, 'blended')
        new.costs()
        new.diagnostics()
        assert before == new.state_digest()
        result['actual_%s_no_shadow_pure' % precision] = True
    direct = V25BankMemory(12, 3, routing='local')
    delayed = V25BankMemory(12, 3, routing='local')
    for h, label in zip(H, labels):
        direct.observe_pair(h, h, int(label))
        bank, prediction = delayed.begin_pair(h, h)
        prior_t = delayed.t
        delayed.apply_feedback(bank, h, target4(label) - prediction)
        assert delayed.t == prior_t
        assert np.array_equal(direct.V, delayed.V)
        assert np.array_equal(direct.centers, delayed.centers)
    result['split_immediate_teaching_equivalence'] = True
    pre = delayed.V.copy()
    bank, prediction = delayed.begin_pair(H[0], H[0])
    delayed.apply_feedback(bank, H[0], .5, action=2)
    diff = delayed.V - pre
    assert not np.any(diff[:, :, [0, 1, 3]])
    assert np.any(diff[bank, :, 2])
    result['reward_action_gate_only'] = True
    isolated = V25BankMemory(12, 11, banks=8)
    spill = V25BankMemory(12, 11, banks=8, spillover=.1, normalize_spill=True)
    a = isolated.observe_pair(H[0], H[0], 1)
    b = spill.observe_pair(H[0], H[0], 1)
    assert math.isclose(a['update_norm'], b['update_norm'], rel_tol=1e-12)
    assert np.count_nonzero(np.linalg.norm(spill.V, axis=(1, 2))) == 8
    result['spill_same_state_proposal_norm_matched'] = True
    for invalid in (dict(protection='neuron', release=True), dict(spillover=.1)):
        guarded = V25BankMemory(12, 0, **invalid)
        try:
            guarded.begin_pair(H[0], H[0])
        except ValueError:
            pass
        else:
            raise AssertionError('Unsupported delayed-teaching mechanism was admitted')
    result['teacher_unsupported_mechanism_guard'] = True
    return result


# === V25 observable task streams and bounded teaching ===


def v25_stream_hash(stream):
    """Hash inputs, targets, ordering and transformed probe controls."""
    h = hashlib.sha256()
    for key in ('train', 'probe', 'original'):
        h.update(np.asarray(stream[key]).tobytes())
    for ph in stream['schedule']:
        h.update(str(ph['name']).encode())
        h.update(np.asarray(ph['targets'], dtype='<i8').tobytes())
        for cp in ph['checkpoints']:
            h.update(np.asarray(cp['events'], dtype='<i8').tobytes())
    for control_name in ('structured_controls', 'context_controls'):
        for key, value in sorted(stream.get(control_name, {}).items()):
            if isinstance(value, np.ndarray):
                h.update(key.encode())
                h.update(value.tobytes())
    return h.hexdigest()


def v25_stationary_labels(stream, prefix):
    """Environment-only conversion of revision phases to same-rule refreshes."""
    labels = np.asarray(stream['original'], dtype=int)
    for ph in stream['schedule']:
        if ph.get('revision', False):
            ph['name'] = prefix + '_refresh_' + ph['name']
        ph['revision'] = False
        ph['ever_revised'] = np.zeros(len(labels), dtype=bool)
        ph['targets'] = labels.copy()
        for cp in ph['checkpoints']:
            cp['events'] = [(int(i), int(v), int(labels[i]))
                            for i, v, _ in cp['events']]
    stream['distinct_input_outcome_pairs'] = int(stream['unique_inputs'])


def v25_erase_order(x):
    out = np.array(x, copy=True)
    average = .5 * (out[..., 0:3, :] + out[..., 5:8, :])
    out[..., 0:3, :] = average
    out[..., 5:8, :] = average
    return out


def v25_context_stream(cfg, seed):
    """A completely specified three-event episode, with no cross-episode state.

    Ticks 0:3 present context, 3:5 an independent distractor, 5:8 a late cue,
    8:12 blanks. Label = 2 * sign(context[0]) + sign(late_cue[0]). Every
    held-out episode has independent continuous nuisance inputs. A second
    held-out view changes only the context sign, preserving the entire late
    cue and distractor, and changes the expected answer accordingly.

    This tests memory across three sensory events in twelve ticks. It is not
    represented as continuous memory across unrelated twelve-tick episodes.
    """
    out = build_stream(cfg, seed)
    labels = np.asarray(out['original'], dtype=int)
    n = len(labels)
    bits = np.stack((labels // 2, labels % 2), axis=1)
    r = rng_for(seed, 'v25_context_latent_train')
    bases = cue_bases(r, n)
    bases[:, :, 0] = (2 * bits - 1) * r.uniform(.75, 1.15, (n, 2))
    out['train'] = render_cues(bases, cfg.variants,
                               rng_for(seed, 'v25_context_train_noise'), cfg.noise)
    train_dist = r.normal(0, .65, (n, 32)).astype(np.float32)
    rd = rng_for(seed, 'v25_context_train_distractor_noise')
    for i in range(n):
        for v in range(cfg.variants):
            out['train'][i, v, 3:5] = train_dist[i] + rd.normal(0, cfg.noise, 32)
    rp = rng_for(seed, 'v25_context_latent_probe')
    pb = cue_bases(rp, n * cfg.probe_variants)
    repeated_bits = np.repeat(bits, cfg.probe_variants, axis=0)
    pb[:, :, 0] = (2 * repeated_bits - 1) * rp.uniform(.75, 1.15, (len(pb), 2))
    out['probe'] = render_cues(pb, 1,
                               rng_for(seed, 'v25_context_probe_noise'), cfg.noise
                               ).reshape(n, cfg.probe_variants, 12, 32)
    pd = rp.normal(0, .65, (n, cfg.probe_variants, 32))
    pd += rp.normal(0, cfg.noise, pd.shape)
    out['probe'][:, :, 3:5] = pd[:, :, None, :]
    flipped = out['probe'].copy()
    flipped[:, :, 0:3, 0] *= -1
    reset = out['probe'].copy()
    reset[:, :, 0:5] = 0
    out['context_controls'] = dict(
        probe_flipped=flipped, targets_flipped=labels ^ 2,
        probe_reset=reset, same_late_cue_and_distractor=True,
        context_reset_rule='Zero ticks 0:5, retain exact late cue ticks 5:8 and trailing blanks.',
        training_identical_late_cue_across_contexts=False,
        probe_identical_late_cue_across_contexts=True)
    v25_stationary_labels(out, 'context')
    out['task_metadata'] = dict(
        task='context_bit_then_distractor_then_late_cue',
        rule='2*sign_category(early_context_coordinate0) + sign_category(late_cue_coordinate0)',
        early_context_ticks=[0, 1, 2], distractor_ticks=[3, 4],
        late_cue_ticks=[5, 6, 7], final_blank_ticks=[8, 9, 10, 11],
        episode_ticks=12, sensory_events_per_episode=3,
        external_labeled_episodes=int(out['external']),
        distinct_sensory_event_presentations=3 * int(out['external']),
        label_messages_per_episode=1, cross_episode_hidden_state=False,
        trial_boundaries='Twelve-tick episodes have explicit resets; no reset between constituent events.',
        probe_context='Every probe starts reset and receives its complete early/distractor/late history.',
        training_episode_identities=n,
        probe_episode_identities=n * cfg.probe_variants,
        train_probe_latent_identity_overlap=0,
        probe_metric='Generalization to unseen latent episodes; seen mask selects exposed rule cohorts.',
        context_absent_balanced_pair_accuracy_ceiling=.5,
        revision_phases='same-rule refresh',
        evaluator_ids_never_learner_inputs=True)
    out['prefix_check'] = dict(status='not_applicable', reason='Context-dependent unseen-episode task')
    out['full_stream_sha256'] = v25_stream_hash(out)
    return out


def v25_prepare_stream(cfg, seed, task='association'):
    if task in ('structured', 'structured_erased', 'structured_order_erased'):
        out = v24_structured_stream(cfg, seed)
        if task != 'structured':
            # V24 transformed training/probes but forgot its swapped probes.
            # Both the ordinary and swapped control must have order erased.
            for key in ('train', 'probe'):
                out[key] = v25_erase_order(out[key])
            out['structured_controls']['probe_swapped'] = v25_erase_order(
                out['structured_controls']['probe_swapped'])
            out['task_metadata']['order_erased'] = True
            out['task_metadata']['erasure_includes_swapped_controls'] = True
        else:
            out['task_metadata']['order_erased'] = False
    elif task in ('context', 'context_reset', 'context_lateonly'):
        out = v25_context_stream(cfg, seed)
        if task != 'context':
            for key in ('train', 'probe'):
                out[key] = np.array(out[key], copy=True)
                out[key][..., 0:5, :] = 0
            for key in ('probe_flipped', 'probe_reset'):
                out['context_controls'][key] = np.array(out['context_controls'][key], copy=True)
                out['context_controls'][key][..., 0:5, :] = 0
            out['task_metadata']['context_intervention'] = 'Erase history before late cue at tick5.'
            out['task_metadata']['context_absent'] = True
            out['task_metadata']['protocol_note'] = (
                'This is a sensory-history erasure control: five blank ticks replace early context '
                'and distractor before the unchanged late cue. It removes context information but '
                'does not claim exact numerical equivalence to resetting a biased recurrent circuit '
                'at tick5. Every episode starts from the prescribed reset state.')
    elif task == 'association':
        out = build_stream(cfg, seed)
        out['task_metadata'] = dict(task='continual_random_association',
                                    sensory_events_per_episode=2,
                                    external_labeled_episodes=int(out['external']),
                                    distinct_sensory_event_presentations=2 * int(out['external']),
                                    probe_metric='Noisy held-out rendering of learned identity')
    else:
        raise ValueError('Unknown V25 task ' + str(task))
    out['task'] = str(task)
    out['full_stream_sha256'] = v25_stream_hash(out)
    return out


def v25_teaching_options(options):
    """Expand a frozen manifest protocol into explicit teacher assumptions."""
    opt = dict(options or {})
    protocol = opt.get('protocol', opt.get('condition', 'immediate'))
    aliases = {'isolated': 'isolated_delay3', 'broadcast': 'intervening_broadcast',
               'known_delay': 'known_delay3', 'exact_delay': 'known_delay3',
               'variable_fifo': 'variable_delay_fifo',
               'variable_broadcast': 'variable_delay_broadcast',
               'existing': 'reward_existing', 'optimistic': 'reward_optimistic'}
    protocol = aliases.get(protocol, protocol)
    if opt.get('mode') == 'reward' and protocol == 'immediate':
        protocol = 'reward_optimistic' if opt.get('optimistic', False) else 'reward_existing'
    allowed = ('immediate', 'isolated_delay3', 'intervening_broadcast', 'known_delay3',
               'variable_delay_fifo', 'variable_delay_broadcast', 'reward_existing', 'reward_optimistic')
    if protocol not in allowed:
        raise ValueError('Unknown V25 teaching protocol ' + str(protocol))
    opt['protocol'] = protocol
    opt['mode'] = 'reward' if protocol.startswith('reward_') else 'target'
    opt['regime'] = 'isolated' if protocol in ('immediate', 'isolated_delay3',
                                              'reward_existing', 'reward_optimistic') else 'intervening'
    opt['delay'] = 3 if protocol in ('isolated_delay3', 'intervening_broadcast', 'known_delay3') else 0
    opt['variable_delay'] = protocol.startswith('variable_delay_')
    opt['selection'] = ('fifo' if protocol == 'variable_delay_fifo' else
                        'known_delay' if protocol == 'known_delay3' else
                        'broadcast' if protocol in ('intervening_broadcast', 'variable_delay_broadcast') else
                        'current')
    opt['retirement'] = ('oldest_pending_after_each_outcome' if opt['variable_delay'] else
                         'retained_until_ring_overwrite' if protocol == 'intervening_broadcast' else
                         'selected_trace_after_feedback')
    opt['optimistic_prior'] = 1.0 if protocol == 'reward_optimistic' else 0.0
    opt.setdefault('gamma', .9)
    opt.setdefault('epsilon', .1)
    opt.setdefault('trace_slots', 32)
    return opt


class V25TeachingState:
    """Bounded physical-site eligibility with no environment return address.

    Learner receives sensory address/value vectors followed by only an outcome
    and its intrinsic clock. Birth times, destinations and action tags arise
    inside the learner. Pending outcomes and identity-based diagnostics stay
    in the evaluator. FIFO is valid only for the explicitly ordered, complete
    feedback channel used here; it is not general variable-delay attribution.
    """
    def __init__(self, model, seed, options):
        self.model = model
        self.options = v25_teaching_options(options)
        self.capacity = int(self.options['trace_slots'])
        self.gamma = float(self.options['gamma'])
        self.epsilon = float(self.options['epsilon'])
        if self.capacity < 1 or not 0 <= self.gamma <= 1 or not 0 <= self.epsilon <= 1:
            raise ValueError('Invalid eligibility capacity, gamma or exploration probability')
        if getattr(model, 'protection', 'none') not in ('none', None) or getattr(model, 'spillover', 0.):
            raise ValueError('V25 teaching is specified only for unprotected isolated bank writes')
        self.prior = float(self.options['optimistic_prior'])
        self.rng = np.random.default_rng(int(seed))
        self.z = np.zeros((self.capacity, int(model.d)), dtype=np.float64)
        self.pred = np.zeros((self.capacity, 4), dtype=np.float64)
        self.destination = np.full(self.capacity, -1, dtype=np.int64)
        self.born = np.full(self.capacity, -1, dtype=np.int64)
        self.action = np.zeros(self.capacity, dtype=np.int64)
        self.present = np.zeros(self.capacity, dtype=bool)
        self.next_slot = 0
        self.events = self.feedback_events = self.exploratory = self.overwritten = 0
        self.missing = self.retired = self.eligibility_applications = 0
        self.reward_sum = self.update_norm_sum = self.mass_sum = self.gain_sum = 0.0
        self.last_selected = ()

    def arrays(self):
        return {k: v for k, v in vars(self).items() if isinstance(v, np.ndarray)}

    def output(self, r, z, view='hard'):
        return np.asarray(self.model.output_pair(r, z, view=view)) + self.prior

    def begin(self, r, z, clock):
        destination, pred = self.model.begin_pair(r, z)
        pred = np.asarray(pred, dtype=np.float64).reshape(4) + self.prior
        exploration = self.options['mode'] == 'reward' and self.rng.random() < self.epsilon
        action = int(self.rng.integers(4)) if exploration else int(np.argmax(pred))
        self.exploratory += int(exploration)
        slot = self.next_slot
        self.overwritten += int(self.present[slot])
        self.z[slot] = np.asarray(z).reshape(-1)
        self.pred[slot] = pred
        self.destination[slot] = destination
        self.born[slot] = int(clock)
        self.action[slot] = action
        self.present[slot] = True
        self.next_slot = (slot + 1) % self.capacity
        self.events += 1
        return action

    def deliver(self, outcome, clock):
        self.feedback_events += 1
        available = np.flatnonzero(self.present & (self.born <= int(clock)))
        if self.options['mode'] == 'reward':
            if float(outcome) not in (0., 1.):
                raise ValueError('Reward teacher receives only chosen-action reward zero or one')
            self.reward_sum += float(outcome)
        if not len(available):
            self.missing += 1
            self.last_selected = ()
            return dict(trace_available=False, eligible_trace_count=0, update_norm=0.)
        oldest = available[np.argmin(self.born[available])]
        selection = self.options['selection']
        if selection == 'fifo':
            selected = np.asarray([oldest], dtype=int)
        elif selection == 'known_delay':
            selected = available[self.born[available] == int(clock) - int(self.options['delay'])]
        elif selection == 'current':
            selected = available[np.asarray([np.argmax(self.born[available])])]
        else:
            selected = available
        self.last_selected = tuple(int(s) for s in selected)
        if not len(selected):
            self.missing += 1
            return dict(trace_available=False, eligible_trace_count=0, update_norm=0.)
        decay = self.gamma ** np.maximum(0, int(clock) - self.born[selected])
        mass = float(decay.sum())
        gain = 1. / max(1., mass)
        # All errors are computed from stored pre-teaching selected-bank values;
        # later feedback neither reroutes the old event nor differentiates a mixture.
        before = self.model.V.copy()
        for slot, weight in zip(selected, decay):
            if self.options['mode'] == 'reward':
                action = int(self.action[slot])
                error = float(outcome) - float(self.pred[slot, action])
            else:
                action = None
                error = target4(int(outcome)) - self.pred[slot]
            self.model.apply_feedback(int(self.destination[slot]), self.z[slot], error,
                                      eta_scale=float(weight) * gain, action=action)
        norm = float(np.linalg.norm(self.model.V - before))
        if self.options['retirement'] == 'oldest_pending_after_each_outcome':
            self.present[oldest] = False
            self.retired += 1
        elif self.options['retirement'] == 'selected_trace_after_feedback':
            self.present[selected] = False
            self.retired += len(selected)
        self.eligibility_applications += len(selected)
        self.update_norm_sum += norm
        self.mass_sum += mass
        self.gain_sum += gain
        return dict(trace_available=True, eligible_trace_count=len(selected),
                    update_norm=norm, trace_mass=mass, feedback_gain=gain)

    def diagnostics(self):
        return dict(protocol=self.options['protocol'], external_events=self.events,
                    feedback_events=self.feedback_events, exploratory_actions=self.exploratory,
                    mean_observed_reward=self.reward_sum / max(1, self.feedback_events)
                    if self.options['mode'] == 'reward' else None,
                    eligibility_slots=self.capacity, overwritten_eligibility_traces=self.overwritten,
                    missing_eligible_feedback=self.missing, retired_eligibility_traces=self.retired,
                    pending_traces=int(self.present.sum()), eligibility_applications=self.eligibility_applications,
                    mean_eligible_trace_count=self.eligibility_applications / max(1, self.feedback_events),
                    mean_update_norm=self.update_norm_sum / max(1, self.feedback_events),
                    mean_trace_mass=self.mass_sum / max(1, self.feedback_events),
                    mean_feedback_gain=self.gain_sum / max(1, self.feedback_events),
                    optimistic_action_prior=self.prior,
                    eligibility_selection=self.options['selection'],
                    eligibility_retirement=self.options['retirement'])


def v25_run_teaching(cfg, job, bundle, stream, make_model):
    """Frozen encoder teaching comparison; no learner-visible phase/identity tags."""
    if job.get('encoder_kind', 'shared_frozen') != 'shared_frozen':
        raise ValueError('Teaching panel requires its declared shared frozen encoder')
    started = time.perf_counter()
    model = make_model()
    teacher = V25TeachingState(model, int(job['seed']) + 90217, job.get('teaching'))
    opt = teacher.options
    result = dict(job=job, status='completed', metrics=[], diagnostics=[], causality=[],
                  costs={}, development={}, stream=dict(stream.get('task_metadata', {})), failures=[])
    shape = stream['train'].shape[:2]
    probe_shape = stream['probe'].shape[:2]
    initial_encoder_costs = dict(bundle.costs())
    encoding_started = time.perf_counter()
    train = np.asarray(bundle.features(stream['train'].reshape(-1, 12, 32), adapt=False)['S'])
    train = train.reshape(*shape, -1)
    probe = np.asarray(bundle.features(stream['probe'].reshape(-1, 12, 32), adapt=False)['S'])
    probe = probe.reshape(*probe_shape, -1)
    encoding_seconds = time.perf_counter() - encoding_started
    if train.shape[-1] != int(model.d) or int(model.d) != 96:
        raise ValueError('Teacher panel expects full-width shared 96-dimensional features')
    queue = []  # Environment holds only due time and target/reward, never a return address.
    delay_rng = rng_for(int(job['seed']), 'v25_teaching_environment_delays')
    clock = external = peak_queue = blank_ticks = messages = 0
    last_due = -1
    correct = first_correct = second_correct = first_n = second_n = 0
    occurrence = {}  # Evaluator-only occurrence/performance ledger; never passed to teacher.
    delay_histogram = {}
    reward_sum = 0.
    evaluation_seconds = 0.

    def receive_due(now):
        nonlocal messages
        while queue and queue[0][0] <= now:
            _, payload = queue.pop(0)
            teacher.deliver(payload, now)
            messages += 1

    for ph in stream['schedule']:
        replacement_trials = {}
        replacement_found = {}
        phase_reward_sum = phase_events = 0
        for cp in ph['checkpoints']:
            for i, v, label in cp['events']:
                if opt['regime'] == 'intervening':
                    receive_due(clock)
                z = train[int(i), int(v)]
                action = teacher.begin(z, z, clock)
                success = int(action == int(label))
                correct += success
                reward_sum += success
                phase_reward_sum += success
                phase_events += 1
                external += 1
                # IDs and changed targets are used here only AFTER the learner action.
                key = (int(i), int(label))
                recurrence = occurrence.get(key, 0)
                occurrence[key] = recurrence + 1
                if recurrence == 0:
                    first_n += 1
                    first_correct += success
                elif recurrence == 1:
                    second_n += 1
                    second_correct += success
                if ph.get('revision', False):
                    replacement_trials[int(i)] = replacement_trials.get(int(i), 0) + 1
                    if success and int(i) not in replacement_found:
                        replacement_found[int(i)] = replacement_trials[int(i)]
                payload = float(success) if opt['mode'] == 'reward' else int(label)
                if opt['variable_delay']:
                    requested = int(delay_rng.integers(2, 6))
                    due = max(clock + requested, last_due)
                    last_due = due
                else:
                    due = clock + int(opt['delay'])
                actual_delay = int(due - clock)
                delay_histogram[actual_delay] = delay_histogram.get(actual_delay, 0) + 1
                queue.append((due, payload))
                peak_queue = max(peak_queue, len(queue))
                if opt['regime'] == 'isolated':
                    for tick in range(int(opt['delay']) + 1):
                        receive_due(clock + tick)
                    clock += int(opt['delay']) + 1
                    blank_ticks += int(opt['delay'])
                else:
                    receive_due(clock)
                    clock += 1
            # Feedback is flushed for equal-received-message evaluation; no phase ID
            # or explicit trace reset is sent. The learner sees ordinary blank ticks.
            while queue:
                receive_due(clock)
                clock += 1
                blank_ticks += 1
            if cp is not ph['checkpoints'][-1]:
                continue
            eval_started = time.perf_counter()
            digest = model.state_digest()
            for view in ('hard', 'blended'):
                scores = teacher.output(probe.reshape(-1, 96), probe.reshape(-1, 96), view=view)
                if not np.isfinite(scores).all():
                    raise FloatingPointError('Nonfinite teaching evaluation')
                row = v24_metric_scores(scores, ph, stream, cfg)
                row.update(phase=ph['name'], revision=bool(ph.get('revision', False)),
                           pass_index=int(cp['pass_index']), external_presentations=external,
                           external=external, simulation_ticks=clock, view=view,
                           prequential_accuracy=correct / max(1, external),
                           first_exposure_accuracy=first_correct / max(1, first_n),
                           next_recurrence_accuracy=second_correct / max(1, second_n),
                           first_exposure_count=first_n, next_recurrence_count=second_n,
                           mean_observed_reward=reward_sum / max(1, external)
                           if opt['mode'] == 'reward' else None,
                           phase_online_accuracy=phase_reward_sum / max(1, phase_events),
                           teaching_protocol=opt['protocol'], teaching_mode=opt['mode'],
                           behavior_policy_view='hard')
                if opt['mode'] == 'reward' and ph.get('revision', False):
                    row['replacement_action_discovery_fraction'] = len(replacement_found) / max(1, len(ph['ids']))
                    row['presentations_to_discovery_among_discovered'] = (
                        float(np.mean(list(replacement_found.values()))) if replacement_found else None)
                    row['replacement_discovery_censored_count'] = len(ph['ids']) - len(replacement_found)
                result['metrics'].append(row)
            if digest != model.state_digest():
                raise AssertionError('Teaching read views changed learner memory')
            evaluation_seconds += time.perf_counter() - eval_started
            diagnostic = teacher.diagnostics()
            diagnostic.update(phase=ph['name'], pass_index=int(cp['pass_index']),
                              blank_simulation_ticks=blank_ticks, pending_environment_queue_peak=peak_queue,
                              evaluator_occurrence_keys=len(occurrence))
            result['diagnostics'].append(diagnostic)
    if messages != external:
        raise AssertionError('Missing/extra outcomes after final flush')
    cost = dict(model.costs())
    eligibility_bytes = sum(a.nbytes for a in teacher.arrays().values())
    encoder_cost = dict(bundle.costs())
    branches = int(encoder_cost.get('encoder_branch_count', 1))
    unique_train_forwards = int(np.prod(shape)) * branches
    unique_probe_forwards = int(np.prod(probe_shape)) * branches
    phase_probe_forwards = unique_probe_forwards * len(stream['schedule'])
    raw_training_forwards = int(encoder_cost.get('encoder_training_branch_evaluations', 0))
    raw_probe_forwards = int(encoder_cost.get('encoder_probe_branch_evaluations', 0))
    # Feature caching is an exact implementation optimization for this frozen
    # sensory branch. Count its conceptual online work for every actual event;
    # retain execution work separately rather than reporting zero learner cost.
    encoder_cost.update(
        encoder_sensory_presentations=external,
        encoder_training_branch_evaluations=external * branches,
        encoder_training_time_bin_evaluations=12 * external * branches,
        encoder_probe_branch_evaluations=phase_probe_forwards,
        encoder_probe_time_bin_evaluations=12 * phase_probe_forwards,
        encoder_total_branch_evaluations=external * branches + phase_probe_forwards,
        encoder_actual_unique_training_cache_branch_evaluations=unique_train_forwards,
        encoder_actual_unique_probe_cache_branch_evaluations=unique_probe_forwards,
        encoder_actual_training_cache_time_bin_evaluations=12 * unique_train_forwards,
        encoder_actual_probe_cache_time_bin_evaluations=12 * unique_probe_forwards,
        encoder_actual_total_cache_branch_evaluations=unique_train_forwards + unique_probe_forwards,
        encoder_actual_bundle_training_branch_counter_delta=raw_training_forwards - int(
            initial_encoder_costs.get('encoder_training_branch_evaluations', 0)),
        encoder_actual_bundle_probe_branch_counter_delta=raw_probe_forwards - int(
            initial_encoder_costs.get('encoder_probe_branch_evaluations', 0)),
        encoder_counting_policy='Online branch/time-bin work charged per external episode; '
            'phase-end probe work charged once per phase, shared across read views; exact frozen '
            'cache executions separately reported as actual unique training/probe cache forwards.',
        encoder_cache_does_not_change_external_presentations=True)
    memory_bytes = int(cost.get('numeric_persistent_bytes', cost.get('persistent_bytes', 0)))
    encoder_bytes = int(encoder_cost.get('encoder_array_bytes', 0))
    prior_bytes = 8
    total_numeric_bytes = memory_bytes + encoder_bytes + eligibility_bytes + prior_bytes
    cost.update(external_presentations=external, labeled_presentations=external if opt['mode'] == 'target' else 0,
                external_presentations_completed=external, total_learning_presentations=external,
                total_labeled_presentations=external if opt['mode'] == 'target' else 0,
                external_sensory_events=external * int(stream.get('task_metadata', {}).get(
                    'sensory_events_per_episode', 2)),
                scalar_reward_presentations=external if opt['mode'] == 'reward' else 0,
                total_teacher_presentations=messages, extra_labeled_presentations=0,
                extra_presentations=0, unique_training_inputs=int(stream['unique_inputs']),
                teacher_messages=messages, teacher_channel_count=1 if opt['mode'] == 'reward' else 4,
                eligibility_mutable_bytes=eligibility_bytes, optimistic_prior_static_bytes=prior_bytes,
                memory_numeric_persistent_bytes=memory_bytes,
                encoder_numeric_persistent_bytes=encoder_bytes,
                eligibility_numeric_persistent_bytes=eligibility_bytes,
                teacher_prior_numeric_persistent_bytes=prior_bytes,
                total_numeric_persistent_bytes=total_numeric_bytes,
                total_numeric_mutable_bytes=int(cost.get('mutable_bytes', 0)) + eligibility_bytes +
                    int(encoder_cost.get('encoder_mutable_array_bytes', 0)),
                total_numeric_fixed_bytes=int(cost.get('fixed_bytes', 0)) + prior_bytes +
                    int(encoder_cost.get('encoder_fixed_array_bytes', encoder_bytes)),
                total_numeric_memory_scope='Model numeric arrays + encoder numeric arrays + bounded '
                    'eligibility arrays + one float64-equivalent fixed reward prior; excludes evaluator '
                    'caches, pending environment outcomes and Python object overhead.',
                encoder=encoder_cost, eligibility_local_update_applications=teacher.eligibility_applications,
                simulation_ticks=clock, blank_simulation_ticks=blank_ticks,
                requested_variable_delay_range=[2, 5] if opt['variable_delay'] else None,
                actual_delay_histogram={str(k): v for k, v in sorted(delay_histogram.items())},
                pending_environment_queue_peak=peak_queue,
                environment_payload_bytes_peak=16 * peak_queue,
                evaluator_feature_cache_bytes=train.nbytes + probe.nbytes,
                evaluator_queue_excluded_from_learner_memory=True,
                external_feedback_fields='Teaching outcome only; intrinsic delivery clock, no return address or identity',
                feedback_order='In order with ties FIFO; no missing outcomes' if opt['variable_delay'] else 'Fixed latency; no missing outcomes',
                attribution_assumption='One outcome per event, ordered complete channel' if opt['variable_delay'] else
                'Exact fixed latency known to selector' if opt['selection'] == 'known_delay' else
                'Current/isolated trial' if opt['selection'] == 'current' else 'Broadcast without return address',
                eligibility_selection=opt['selection'], eligibility_retirement=opt['retirement'],
                feedback_gain_rule='gamma^age / max(1,sum eligible gamma^age); delayed traces are not renormalized to erase attenuation',
                teaching_prediction='Selected bank pre-update scores; reward adds declared fixed action prior',
                behavior_policy_view='hard; blended terminal scores are an alternate read view, not an independently trained behavior policy',
                reward_exploration='epsilon=.1 unchanged; optimistic alternative adds fixed score prior1 before selected-action reward delta',
                protection_or_spillover_allowed=False)
    result['costs'] = cost
    result['timing'] = dict(total_seconds=time.perf_counter() - started,
                            encoding_seconds=encoding_seconds, evaluation_seconds=evaluation_seconds)
    return result


def v25_task_control_rows(model, bundle, stream, job, ph, cfg):
    """Evaluator-only paired interventions, at root-selected checkpoints.

    Uses current model weights and current encoder without adapting either.
    IDs determine evaluation masks only. Control histories are complete inputs,
    with the same interventions applied to ordinary and flipped/swapped views.
    """
    controls = stream.get('structured_controls') or stream.get('context_controls')
    if not controls:
        return []
    path = job.get('path', 'SS')
    address_key, value_key = path[0], path[1]
    width = int(job.get('route_dims', 96))
    digest = model.state_digest()

    def encode_pair(raw):
        feats = bundle.features(np.asarray(raw).reshape(-1, 12, 32), adapt=False)
        r = np.asarray(feats[address_key])[:, :width]
        z = np.asarray(feats[value_key])
        return r, z

    base_r, base_z = encode_pair(stream['probe'])
    rows = []
    seen = np.asarray(ph['seen'], dtype=bool)
    if 'probe_swapped' in controls:
        control_r, control_z = encode_pair(controls['probe_swapped'])
        control_targets = np.asarray(controls['targets_swapped'], dtype=int)
        sensitive = seen & np.asarray(controls['order_sensitive_mask'], dtype=bool)
        for view in ('hard', 'blended'):
            normal = np.asarray(model.output_pair(base_r, base_z, view=view))
            swapped = np.asarray(model.output_pair(control_r, control_z, view=view))
            normal_pred = normal.reshape(len(seen), cfg.probe_variants, 4).argmax(-1)
            swapped_pred = swapped.reshape(len(seen), cfg.probe_variants, 4).argmax(-1)
            rows.append(dict(
                diagnostic_kind='structured_order_control', phase=ph['name'], view=view,
                order_erased=bool(stream.get('task_metadata', {}).get('order_erased', False)),
                normal_unseen_accuracy=accuracy(normal, ph['targets'], seen, cfg.probe_variants),
                swapped_unseen_accuracy=accuracy(swapped, control_targets, seen, cfg.probe_variants),
                swapped_order_sensitive_accuracy=accuracy(swapped, control_targets, sensitive, cfg.probe_variants),
                prediction_change_on_sensitive_pairs=float(np.mean(
                    normal_pred[sensitive] != swapped_pred[sensitive])) if sensitive.any() else None,
                paired_both_correct=float(np.mean(
                    (normal_pred[seen] == np.asarray(ph['targets'])[seen, None]) &
                    (swapped_pred[seen] == control_targets[seen, None]))) if seen.any() else None,
                readout_state_unchanged=True))
    else:
        flipped_r, flipped_z = encode_pair(controls['probe_flipped'])
        reset_r, reset_z = encode_pair(controls['probe_reset'])
        flipped_targets = np.asarray(controls['targets_flipped'], dtype=int)
        for view in ('hard', 'blended'):
            normal = np.asarray(model.output_pair(base_r, base_z, view=view))
            flipped = np.asarray(model.output_pair(flipped_r, flipped_z, view=view))
            reset = np.asarray(model.output_pair(reset_r, reset_z, view=view))
            normal_pred = normal.reshape(len(seen), cfg.probe_variants, 4).argmax(-1)
            flipped_pred = flipped.reshape(len(seen), cfg.probe_variants, 4).argmax(-1)
            a0 = accuracy(normal, ph['targets'], seen, cfg.probe_variants)
            a1 = accuracy(flipped, flipped_targets, seen, cfg.probe_variants)
            r0 = accuracy(reset, ph['targets'], seen, cfg.probe_variants)
            r1 = accuracy(reset, flipped_targets, seen, cfg.probe_variants)
            rows.append(dict(
                diagnostic_kind='paired_context_control', phase=ph['name'], view=view,
                context_absent=bool(stream.get('task_metadata', {}).get('context_absent', False)),
                normal_unseen_accuracy=a0, flipped_context_unseen_accuracy=a1,
                balanced_context_accuracy=.5 * (a0 + a1) if a0 is not None else None,
                reset_balanced_context_accuracy=.5 * (r0 + r1) if r0 is not None else None,
                reset_balanced_context_ceiling=.5,
                paired_both_correct=float(np.mean(
                    (normal_pred[seen] == np.asarray(ph['targets'])[seen, None]) &
                    (flipped_pred[seen] == flipped_targets[seen, None]))) if seen.any() else None,
                changed_prediction_with_identical_late_cue=float(np.mean(
                    normal_pred[seen] != flipped_pred[seen])) if seen.any() else None,
                identical_late_cue_and_distractor=True, readout_state_unchanged=True))
    if digest != model.state_digest():
        raise AssertionError('Task control evaluation changed bank memory')
    return rows


def v25_task_checks(cfg):
    report = {}
    structured = v25_prepare_stream(cfg, 991, 'structured_erased')
    assert np.array_equal(structured['probe'], structured['structured_controls']['probe_swapped'])
    assert np.array_equal(structured['probe'][..., 0:3, :], structured['probe'][..., 5:8, :])
    report['order_erasure_applies_to_swapped_probe'] = True
    context = v25_prepare_stream(cfg, 991, 'context')
    controls = context['context_controls']
    assert np.array_equal(context['probe'][..., 3:, :], controls['probe_flipped'][..., 3:, :])
    assert np.array_equal(controls['targets_flipped'], np.asarray(context['original']) ^ 2)
    reset = v25_prepare_stream(cfg, 991, 'context_reset')
    assert not np.any(reset['train'][..., 0:5, :])
    assert np.array_equal(reset['probe'][..., 5:, :], context['probe'][..., 5:, :])
    assert np.array_equal(reset['probe'], reset['context_controls']['probe_flipped'])
    assert all(not ph['revision'] for ph in context['schedule'])
    assert context['task_metadata']['distinct_sensory_event_presentations'] == 3 * context['external']
    report['context_reset_removes_early_bit_preserves_exact_late_cue'] = True
    report['paired_context_probe_changes_only_early_signal'] = True
    report['context_is_three_events_per_labeled_episode'] = True
    return report


def v25_teaching_checks():
    report = {}
    d = 4
    h = np.eye(d)[0]
    factory = lambda: V25BankMemory(d, 123, banks=1)
    ref = factory()
    ref.observe_pair(h, h, 2)
    m = factory()
    state = V25TeachingState(m, 77, {'protocol': 'immediate'})
    state.begin(h, h, 0)
    state.deliver(2, 0)
    assert np.allclose(ref.V, m.V)
    assert not state.present.any()
    report['immediate_target_matches_selected_bank_observe'] = True
    m = factory()
    state = V25TeachingState(m, 77, {'protocol': 'known_delay3'})
    state.begin(h, h, 0)
    state.begin(np.eye(d)[1], np.eye(d)[1], 1)
    state.deliver(2, 3)
    assert np.allclose(m.V[0, 0], ref.V[0, 0] * .9 ** 3)
    assert not np.any(m.V[0, 1])
    report['known_delay_selects_own_birth_time_and_preserves_decay'] = True
    m = factory()
    state = V25TeachingState(m, 77, {'protocol': 'variable_delay_fifo'})
    state.begin(h, h, 0)
    state.begin(np.eye(d)[1], np.eye(d)[1], 1)
    state.deliver(2, 4)
    assert state.last_selected == (0,) and not state.present[0] and state.present[1]
    state.deliver(3, 4)
    assert state.last_selected == (1,) and not state.present.any()
    assert np.any(m.V[0, 0]) and np.any(m.V[0, 1])
    report['fifo_tied_arrivals_select_and_retire_in_order'] = True
    m = factory()
    state = V25TeachingState(m, 77, {'protocol': 'variable_delay_broadcast'})
    state.begin(h, h, 0)
    state.begin(np.eye(d)[1], np.eye(d)[1], 1)
    state.deliver(2, 4)
    assert state.last_selected == (0, 1) and not state.present[0] and state.present[1]
    assert np.any(m.V[0, 0]) and np.any(m.V[0, 1])
    report['variable_broadcast_same_pending_lifetime_as_fifo'] = True
    m = factory()
    state = V25TeachingState(m, 77, {'protocol': 'reward_optimistic', 'epsilon': 0.})
    assert np.all(state.output(h, h) == 1.)
    a = state.begin(h, h, 0)
    state.deliver(0., 0)
    changed_outputs = np.flatnonzero(np.any(m.V != 0, axis=(0, 1)))
    assert changed_outputs.tolist() == [a]
    assert np.argmax(state.output(h, h)) != a
    report['optimistic_prior_independent_of_signed_input_and_chosen_action_only'] = True
    return report


# V25 predeclared roster. No score-dependent selection.
V25_VERSION = '25.0.0'
V25_SEEDS = (251, 277, 307, 331, 359, 383)
V25_PANELS = ('causal','practical','reference','encoder','precision','placement',
              'structured','context','teaching','reward')

def v25_jobs(cfg):
    jobs=[]
    for seed in cfg.seeds:
        byname={}
        def add(panel,name,kind='dual',path='FF',rate='base',routing='fixed',
                route_dims=96,task='association',protection=False,precision='float64',
                spill=0.,normalize=False,teaching=None,parent=None,contrast=None,gate=None,banks=8):
            eta=cfg.learning_rate*(.5 if rate=='slow' else 1.)
            memory=dict(banks=banks,routing=routing,eta=eta,
                protection='neuron' if protection else 'none',release=bool(protection),
                precision=precision,spillover=spill,normalize_spill=normalize)
            config=dict(panel=panel,name=name,encoder_kind=kind,path=path,
                route_dims=route_dims,memory=memory,task=task,teaching=teaching,
                locality='no_hidden_backprop_with_declared_population_primitives')
            config_id=sha_bytes(json.dumps(config,sort_keys=True,separators=(',',':')).encode())[:16]
            job=dict(config,config_id=config_id,seed=int(seed),id='%s.s%d.%s.%s'%(panel,seed,name,config_id[:10]),comparisons=[])
            if parent:
                if parent not in byname: raise AssertionError('Parent must precede child '+parent)
                job['comparisons'].append(dict(name=contrast or 'prespecified',parent_id=byname[parent]['id'],
                    margin=.05,gate=gate or 'plasticity'))
                if byname[parent]['memory']['banks']==1:
                    job['comparisons'][-1]['parent_view']='hard'
            jobs.append(job);byname[name]=job
            return job
        # Eight core configurations: fixed prototypes, equal duplicate-encoder cost.
        for rate in ('base','slow'):
            for path in ('FF','LL','FL','LF'):
                name='dual_'+path+'_'+rate
                add('causal',name,path=path,rate=rate,
                    parent=('dual_FF_'+rate if path!='FF' else ('dual_FF_base' if rate=='slow' else None)),
                    contrast='pathway_stability' if path!='FF' else 'write_rate')
        for path in ('LL','FL'):
            for rate in ('base','slow'):
                add('causal','dual_'+path+'_'+rate+'_release',path=path,rate=rate,protection=True,
                    parent='dual_'+path+'_'+rate,contrast='neuron_release')
        for rate in ('base','slow'):
            add('causal','dual_FL_'+rate+'_local_router',path='FL',rate=rate,routing='local',
                parent='dual_FL_'+rate,contrast='router_learning')
        # Matched single-encoder partition, identical 48-coordinate address subset.
        for kind in ('split_frozen','split_live','split_stable'):
            for rate in ('base','slow'):
                add('practical',kind+'_'+rate,kind=kind,path='SS',rate=rate,route_dims=48,
                    parent=('split_live_'+rate if kind=='split_stable' else
                            ('split_frozen_'+rate if kind=='split_live' else None)),contrast='stable_half')
        for rate in ('base','slow'):
            add('practical','split_stable_'+rate+'_local',kind='split_stable',path='SS',rate=rate,
                route_dims=48,routing='local',parent='split_stable_'+rate,contrast='router_learning')
        for rate in ('base','slow'):
            add('practical','split_stable_'+rate+'_release',kind='split_stable',path='SS',rate=rate,
                route_dims=48,routing='local',protection=True,parent='split_stable_'+rate+'_local',contrast='neuron_release')
        add('practical','split_stable_slow_gated',kind='split_stable_gated',path='SS',rate='slow',
            route_dims=48,routing='local',parent='split_stable_slow_local',contrast='sensory_novelty_gate')
        for kind in ('shared_frozen','shared_live'):
            add('reference',kind+'_ordinary',kind=kind,path='SS',banks=1,
                parent='shared_frozen_ordinary' if kind=='shared_live' else None,contrast='encoder_adaptation')
        for rate in ('base','slow'):
            add('reference','e3_local_'+rate,kind='shared_frozen',path='SS',rate=rate,routing='local',
                parent='shared_frozen_ordinary' if rate=='base' else 'e3_local_base',contrast='storage' if rate=='base' else 'write_rate')
        for rate in ('base','slow'):
            add('encoder','minimal_'+rate,kind='minimal',path='SS',rate=rate,routing='local',
                parent='e3_local_'+rate,contrast='minimal_temporal_encoder',gate='minimal_encoder')
        for rate in ('base','slow'):
            add('precision','local_'+rate+'_int8',kind='shared_frozen',path='SS',rate=rate,routing='local',precision='int8',
                parent='e3_local_'+rate,contrast='stored_readout_precision')
        add('placement','local_spill10',kind='shared_frozen',path='SS',routing='local',spill=.1,
            parent='e3_local_base',contrast='spillover_unmatched_strength')
        add('placement','local_spill10_norm',kind='shared_frozen',path='SS',routing='local',spill=.1,normalize=True,
            parent='e3_local_base',contrast='spillover_norm_matched')
        # Structured models are predetermined, never selected by association scores.
        rows=[('structured_ordinary','shared_frozen',1,'fixed','base',96),
              ('structured_fixed','shared_frozen',8,'fixed','base',96),
              ('structured_local','shared_frozen',8,'local','base',96),
              ('structured_slow','shared_frozen',8,'local','slow',96),
              ('structured_stable','split_stable_gated',8,'local','slow',48),
              ('structured_minimal','minimal',8,'local','base',96),
              ('structured_observable','observable',1,'fixed','base',96)]
        for name,kind,banks,routing,rate,dims in rows:
            parent=('structured_local' if kind=='minimal' else
                    ('structured_ordinary' if name not in ('structured_ordinary',) else None))
            add('structured',name,kind=kind,path='SS',banks=banks,routing=routing,rate=rate,route_dims=dims,
                task='structured',parent=parent,contrast='minimal_temporal_encoder' if kind=='minimal' else 'structured_transfer',
                gate='minimal_encoder' if kind=='minimal' else 'none')
        add('structured','structured_fixed_erased',kind='shared_frozen',path='SS',task='structured_erased',
            parent='structured_fixed',contrast='order_erasure',gate='none')
        for name,kind,dims,task in [('context_e3','shared_frozen',96,'context'),
                ('context_stable','split_stable_gated',48,'context'),
                ('context_minimal','minimal',96,'context'),
                ('context_reset','shared_context_reset',96,'context'),
                ('context_observable','observable',96,'context')]:
            add('context',name,kind=kind,path='SS',route_dims=dims,task=task,
                banks=1 if kind=='observable' else 8,routing='fixed',
                parent='context_e3' if name!='context_e3' else None,contrast='context_bridge',gate='none')
        protocols=('immediate','isolated','broadcast','known_delay','variable_fifo','variable_broadcast')
        for protocol in protocols:
            options=dict(mode='target',protocol=protocol,delay=0 if protocol=='immediate' else 3,
                         trace_slots=32,gamma=.9,epsilon=.1,optimistic=False)
            add('teaching','target_'+protocol,kind='shared_frozen',path='SS',teaching=options,
                parent='target_immediate' if protocol!='immediate' else None,contrast='temporal_attribution')
        for protocol in ('immediate','variable_fifo'):
            options=dict(mode='target',protocol=protocol,delay=0 if protocol=='immediate' else 3,
                         trace_slots=32,gamma=.9,epsilon=.1,optimistic=False)
            add('teaching','target_slow_'+protocol,kind='shared_frozen',path='SS',rate='slow',teaching=options,
                parent='target_'+protocol,contrast='delayed_write_rate')
        for optimistic in (False,True):
            options=dict(mode='reward',protocol='immediate',delay=0,trace_slots=32,gamma=.9,epsilon=.1,optimistic=optimistic)
            add('reward','reward_optimistic' if optimistic else 'reward_existing',kind='shared_frozen',path='SS',teaching=options,
                parent='reward_existing' if optimistic else None,contrast='reward_exploration')
        for rate in ('base','slow'):
            for path in ('FL','LF'):
                byname['dual_'+path+'_'+rate]['comparisons'].append(dict(name='rescue_from_live',
                    parent_id=byname['dual_LL_'+rate]['id'],margin=.05,gate='plasticity'))
        for job in byname.values():
            if job['memory']['banks']>1:
                job['comparisons'].append(dict(name='read_expression',parent_id=job['id'],
                    view='blended',parent_view='hard',gate=False))
    selected=[j for j in jobs if ('all' in cfg.panels or j['panel'] in cfg.panels)
              and (not cfg.arms or j['name'] in cfg.arms)]
    if len({j['id'] for j in selected})!=len(selected):raise AssertionError('Duplicate job identity')
    return selected

def v25_design_policy():
    return dict(primary='Separate address coordinates, value coordinates, router adaptation and bank writes.',
        primary_seed_count=6,locality='No hidden backprop; prescribed microdynamics and explicit population normalization.',
        plasticity_margin_pp=5.,minimal_encoder_margin_pp=5.,
        minimal_encoder_decision='Must meet retention and structured-transfer noninferiority separately; six-seed intervals unadjusted.',
        selection='All candidate and task choices fixed before results; no score-dependent expansion.',
        main_stream_presentations=2304,development='E3 local sensory development64 identities x6 passes; calibration64.',
        cache='Sensory-only trajectories reused as evaluator computation; full simulated learner state/operations charged to every arm.',
        duplicate_encoder='Both copies processed and retained in every dual arm, including unused branch.',
        context='Three explicit sensory events within one12-tick episode, not unsegmented cross-episode persistence.',
        deferred=['large scaling','TinyShakespeare','hidden route dictionaries','periodic re-indexing','pair protection',
                  'flawed eviction','allocator sweep','legacy categorical-admission reward transplant'],
        precision='Int8 applies only to stored readout; transient dequantization and encoder costs reported separately.')


# V25 reporting: exact configuration identities and explicitly paired seeds.
import csv as _v25_csv
import gzip as _v25_gzip
import hashlib as _v25_hashlib
import json as _v25_json
import math as _v25_math
import statistics as _v25_statistics
import zipfile as _v25_zipfile
from pathlib import Path as _V25Path


def v25_report_plain(value):
    if isinstance(value, dict):
        return {str(k): v25_report_plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [v25_report_plain(v) for v in value]
    if isinstance(value, _V25Path):
        return str(value)
    if isinstance(value, float):
        return value if _v25_math.isfinite(value) else None
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if hasattr(value, 'item'):
        try:
            return v25_report_plain(value.item())
        except (ValueError, TypeError):
            pass
    if hasattr(value, 'tolist'):
        return v25_report_plain(value.tolist())
    if hasattr(value, '__dict__'):
        return v25_report_plain(vars(value))
    return str(value)


def v25_report_canonical(value):
    return _v25_json.dumps(v25_report_plain(value), sort_keys=True,
                           separators=(',', ':'), allow_nan=False)


def v25_report_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and _v25_math.isfinite(float(value))


def v25_report_interval(values):
    values = [float(x) for x in values if v25_report_number(x)]
    n = len(values)
    mean = _v25_statistics.fmean(values) if n else None
    answer = dict(n=n, mean=mean, sd=None, ci95_low=None, ci95_high=None)
    if n < 2:
        return answer
    criticals = {1: 12.706204736, 2: 4.302652730, 3: 3.182446305,
                 4: 2.776445105, 5: 2.570581836, 6: 2.446911851,
                 7: 2.364624252, 8: 2.306004135, 9: 2.262157163,
                 10: 2.228138852, 12: 2.178812830, 15: 2.131449546,
                 20: 2.085963447, 30: 2.042272456, 60: 2.000297822}
    critical = criticals[max(k for k in criticals if k <= n - 1)]
    sd = _v25_statistics.stdev(values)
    half = critical * sd / _v25_math.sqrt(n)
    answer.update(sd=sd, ci95_low=mean-half, ci95_high=mean+half)
    return answer


def v25_report_configuration(job):
    # Keep every actual override, including all nested memory/encoder settings.
    # Only replicate identifiers and comparison bookkeeping are excluded.
    excluded = {'id', 'config_id', 'seed', 'comparisons', 'parent', 'parent_id', 'parents',
                'parent_keys', 'comparison', 'contrast', 'gate'}
    return {k: v for k, v in job.items() if k not in excluded}


def v25_report_meta(job):
    configuration = v25_report_configuration(job)
    identity = job.get('config_id') or _v25_hashlib.sha256(v25_report_canonical(configuration).encode()).hexdigest()[:20]
    return dict(job_id=job.get('id'), config_id=identity, seed=job.get('seed'),
                panel=job.get('panel'), name=job.get('name'),
                encoder_kind=job.get('encoder_kind'), path=job.get('path'),
                route_dims=job.get('route_dims'), task=job.get('task', 'association'),
                memory=job.get('memory', {}), teaching=job.get('teaching', {}),
                configuration=configuration)


def v25_report_flatten(value, prefix=''):
    out = {}
    for key, val in (value or {}).items():
        key = prefix + str(key)
        if isinstance(val, dict):
            out.update(v25_report_flatten(val, key + '__'))
        else:
            out[key] = v25_report_plain(val)
    return out


def v25_report_phase_rows(rows):
    phases = {}
    for index, raw in enumerate(rows):
        if not isinstance(raw, dict):
            continue
        row = dict(raw, _index=index)
        phase = str(row.get('phase', row.get('checkpoint', 'unknown')))
        key = (row.get('external_presentations') or 0,
               row.get('pass_index') or 0, index)
        if phase not in phases or key >= phases[phase][0]:
            phases[phase] = (key, row)
    return [dict(row, phase=phase) for phase, (key, row) in
            sorted(phases.items(), key=lambda item: item[1][0])]


def v25_report_is_revision(row):
    if 'revision' in row:
        return row['revision'] is True or row['revision'] == 1
    phase = str(row.get('phase', ''))
    return phase.startswith(('revise', 'revision'))


def v25_report_auc(rows, key='all_current'):
    # This is a trapezoidal checkpoint-curve area over the observed span.
    # No unobserved t=0 performance or prequential predictions are invented.
    points = {}
    for row in rows:
        x, y = row.get('external_presentations'), row.get(key)
        if v25_report_number(x) and v25_report_number(y):
            points[float(x)] = float(y)
    ordered = sorted(points.items())
    if len(ordered) < 2:
        return dict(value=None, start=ordered[0][0] if ordered else None,
                    end=ordered[-1][0] if ordered else None, checkpoints=len(ordered))
    area = sum((b-a)*(ya+yb)/2 for (a, ya), (b, yb) in zip(ordered, ordered[1:]))
    span = ordered[-1][0] - ordered[0][0]
    return dict(value=area/span if span else None, start=ordered[0][0],
                end=ordered[-1][0], checkpoints=len(ordered))


def v25_report_summaries(job, result):
    meta = v25_report_meta(job)
    by_view = {}
    for row in result.get('metrics') or []:
        if isinstance(row, dict):
            by_view.setdefault(str(row.get('view', 'hard')), []).append(row)
    if not by_view:
        by_view = {'hard': []}
    summaries = []
    for view, raw_rows in by_view.items():
        rows = v25_report_phase_rows(raw_rows)
        terminal = rows[-1] if rows else {}
        acquisitions = [r for r in rows if not v25_report_is_revision(r)][-3:]
        revisions = [r for r in rows if v25_report_is_revision(r)][-3:]
        row = dict(meta, view=view, status=result.get('status', 'completed'),
                   phases=len(rows), metric_rows=len(raw_rows),
                   terminal_phase=terminal.get('phase'),
                   late_acquisition_phases=len(acquisitions), late_revision_phases=len(revisions),
                   late_acquisition_phase_names=[r['phase'] for r in acquisitions],
                   late_revision_phase_names=[r['phase'] for r in revisions],
                   late_acquisition_short_window=len(acquisitions) < 3,
                   late_revision_short_window=len(revisions) < 3)
        if result.get('failures'):
            row['status'] = 'failed'
        if not raw_rows and row['status'] == 'completed':
            row['status'] = 'no_metrics'
        for key, value in terminal.items():
            if key not in ('_index', 'revision') and (v25_report_number(value) or value is None):
                row['terminal_' + key] = value
        # Explicit transformed-probe endpoints belong to the same trained view.
        # Retain them in the seed summary so model comparisons cannot pass an
        # order-sensitive architecture gate using normal accuracy alone.
        for control in result.get('controls') or []:
            if not isinstance(control, dict) or str(control.get('view', 'hard')) != view:
                continue
            if not control.get('control'):
                continue
            if control.get('phase') is not None and control['phase'] != terminal.get('phase'):
                continue
            for key, value in control.items():
                if key not in ('phase', 'external_presentations') and (v25_report_number(value) or value is None):
                    row['terminal_control__' + str(control['control']) + '__' + key] = value
        for name, key, phase_rows in [('late_acquisition', 'phase_current', acquisitions),
                                      ('late_revision', 'phase_current', revisions),
                                      ('late_retention', 'prior_current', acquisitions)]:
            row[name] = v25_report_interval([r.get(key) for r in phase_rows])['mean']
        auc = v25_report_auc(rows)
        row.update(learning_curve_auc=auc['value'], learning_curve_auc_start=auc['start'],
                   learning_curve_auc_end=auc['end'], learning_curve_auc_checkpoints=auc['checkpoints'])
        for prefix, block in [('cost__', 'costs'), ('time__', 'timing')]:
            if isinstance(result.get(block), dict):
                row.update(v25_report_flatten(result[block], prefix))
        summaries.append(row)
    return summaries


def v25_report_endpoint_keys(rows):
    return sorted({key for row in rows for key, value in row.items()
                   if (key.startswith('terminal_') or key.startswith(('cost__', 'time__'))
                       or key in ('late_acquisition', 'late_revision', 'late_retention', 'learning_curve_auc'))
                   and v25_report_number(value)
                   and key not in ('terminal_external_presentations', 'terminal_pass_index')})


def v25_report_delta_is_fraction(key):
    if key.startswith(('cost__', 'time__')):
        return False
    # Default to native units: counters and latency metrics can have arbitrary
    # names, so a blacklist would silently convert new measurements into pp.
    name = key[len('terminal_'):] if key.startswith('terminal_') else key
    name = name.rsplit('__', 1)[-1]
    fractions = {'all_current', 'A_current', 'B_current', 'C_current', 'initial_revised',
                 'original_never_revised', 'phase_current', 'prior_current', 'initial_unchanged',
                 'late_acquisition', 'late_revision', 'late_retention', 'learning_curve_auc',
                 'mean_observed_reward', 'paired_both_correct',
                 'prediction_change_on_sensitive_pairs', 'changed_prediction_with_identical_late_cue',
                 'reset_balanced_context_ceiling'}
    return name in fractions or name == 'accuracy' or name.endswith(('_accuracy', '_fraction', '_probability'))


def v25_report_aggregate(summaries):
    grouped = {}
    for row in summaries:
        grouped.setdefault((row['config_id'], row['view']), []).append(row)
    aggregates = []
    for (_, view), rows in sorted(grouped.items()):
        good = [r for r in rows if r['status'] in ('completed', 'complete', 'ok')]
        by_seed = {}
        for row in good:
            by_seed.setdefault(row['seed'], row)
        good = list(by_seed.values())
        value = {k: v for k, v in rows[0].items() if k in
                 ('config_id', 'panel', 'name', 'encoder_kind', 'path', 'route_dims',
                  'task', 'memory', 'teaching', 'configuration')}
        value.update(view=view, completed_seeds=len(good), recorded_jobs=len(rows),
                     duplicate_seed_records=len([r for r in rows if r['status'] == 'completed'])-len(good),
                     seeds=sorted(by_seed, key=str))
        for key in v25_report_endpoint_keys(good):
            for label, item in v25_report_interval([r.get(key) for r in good]).items():
                value[key + '_' + label] = item
        aggregates.append(value)
    return aggregates


def v25_report_gate(stats, endpoints, margin):
    relevant = [stats.get(key, {}) for key in endpoints]
    enough = relevant and all(item.get('n', 0) >= 2 and item.get('ci95_low') is not None for item in relevant)
    if enough and all(item['ci95_low'] >= -margin for item in relevant):
        return 'passes_margin'
    if enough and any(item['ci95_high'] < -margin for item in relevant):
        return 'fails_margin'
    return 'inconclusive'


def v25_report_diagnostic_aggregate(rows):
    """Aggregate matched diagnostic checkpoints, never treating probes as seeds."""
    grouped = {}
    ignored = {'job_id', 'seed', 'configuration', 'memory', 'teaching'}
    checkpoint_keys = {'external_presentations', 'checkpoint', 'phase_index', 'pass_index',
                       'from_external_presentations', 'to_external_presentations'}
    for row in rows:
        dimensions = {k: v for k, v in row.items() if k not in ignored and
                      (isinstance(v, (str, bool)) or k in checkpoint_keys)}
        key = v25_report_canonical(dimensions)
        grouped.setdefault(key, []).append(row)
    output = []
    for key, values in sorted(grouped.items()):
        group = _v25_json.loads(key)
        by_seed = {}
        for row in values:
            by_seed.setdefault(row['seed'], []).append(row)
        group.update(seed_count=len(by_seed), seeds=sorted(by_seed, key=str), source_rows=len(values))
        numeric_keys = {k for row in values for k, v in row.items()
                        if k not in ignored and k not in checkpoint_keys and v25_report_number(v)}
        for metric in sorted(numeric_keys):
            # Any repeated records are first reduced within the independent seed.
            means = [v25_report_interval([r.get(metric) for r in rs])['mean'] for rs in by_seed.values()]
            for name, value in v25_report_interval(means).items():
                group[metric+'_'+name] = value
        output.append(group)
    return output


def v25_report_comparisons(jobs, summaries):
    jobs_by_id = {job['id']: job for job in jobs}
    planned_seeds = {}
    for job in jobs:
        planned_seeds.setdefault(v25_report_meta(job)['config_id'], set()).add(job['seed'])
    indexed = {(row['job_id'], row['view']): row for row in summaries}
    pairs, missing = [], []
    for row in summaries:
        job = jobs_by_id[row['job_id']]
        for declaration in job.get('comparisons') or []:
            if declaration.get('view') is not None and str(declaration['view']) != row['view']:
                continue
            parent_id = declaration.get('parent_id')
            parent_view = declaration.get('parent_view', row['view'])
            parent = indexed.get((parent_id, parent_view))
            reason = None
            if parent is None:
                reason = 'declared parent job/view not available'
            elif parent['seed'] != row['seed']:
                reason = 'different seeds cannot form a paired observation'
            elif any(r['status'] not in ('completed', 'complete', 'ok') for r in (row, parent)):
                reason = 'a paired job is incomplete or failed'
            if reason:
                missing.append(dict(job_id=row['job_id'], parent_id=parent_id, view=row['view'],
                                    parent_view=parent_view, name=declaration.get('name'), reason=reason))
                continue
            margin = declaration.get('margin', .05)
            if not v25_report_number(margin) or margin < 0 or margin > 1:
                missing.append(dict(job_id=row['job_id'], parent_id=parent_id,
                                    reason='margin must be a nonnegative accuracy fraction at most 1'))
                continue
            pair = dict(job_id=row['job_id'], parent_id=parent_id, config_id=row['config_id'],
                        parent_config_id=parent['config_id'], name=declaration.get('name', 'declared_comparison'),
                        seed=row['seed'], view=row['view'], parent_view=parent_view, task=row['task'],
                        parent_task=parent['task'], panel=row['panel'], margin=margin,
                        gate_type=declaration.get('gate', 'plasticity'),
                        gate_endpoints=declaration.get('endpoints'),
                        child_configuration=row['configuration'], parent_configuration=parent['configuration'],
                        within_trained_state=parent_id == row['job_id'])
            for key in set(v25_report_endpoint_keys([row])) & set(v25_report_endpoint_keys([parent])):
                if v25_report_number(row.get(key)) and v25_report_number(parent.get(key)):
                    suffix = '_delta_pp' if v25_report_delta_is_fraction(key) else '_delta'
                    pair[key + suffix] = (row[key]-parent[key]) * (1 if suffix == '_delta' else 100)
            pairs.append(pair)
    grouped = {}
    group_fields = ('config_id', 'parent_config_id', 'name', 'view', 'parent_view',
                    'task', 'parent_task', 'panel', 'margin', 'gate_type', 'gate_endpoints', 'within_trained_state')
    for pair in pairs:
        key = v25_report_canonical({k: pair[k] for k in group_fields})
        grouped.setdefault(key, []).append(pair)
    aggregates, gates = [], []
    for key, rows in sorted(grouped.items()):
        group = _v25_json.loads(key)
        by_seed = {}
        for row in rows:
            by_seed.setdefault(row['seed'], row)
        distinct = list(by_seed.values())
        group.update(paired_seeds=len(distinct), seeds=sorted(by_seed, key=str),
                     duplicate_seed_records=len(rows)-len(distinct),
                     child_configuration=rows[0]['child_configuration'],
                     parent_configuration=rows[0]['parent_configuration'])
        expected = planned_seeds.get(group['config_id'], set()) & planned_seeds.get(group['parent_config_id'], set())
        group['planned_paired_seeds'] = len(expected)
        group['planned_pairing_complete'] = bool(expected) and set(by_seed) == expected
        stats = {}
        for endpoint in sorted({k for r in distinct for k in r if k.endswith(('_delta', '_delta_pp'))}):
            stat = v25_report_interval([r.get(endpoint) for r in distinct])
            stats[endpoint] = stat
            for label, value in stat.items():
                group[endpoint + '_' + label] = value
        aggregates.append(group)
        gate_type = group['gate_type']
        if gate_type is False or gate_type in ('none', None):
            continue
        if gate_type == 'minimal_encoder':
            default_endpoints = ['terminal_all_current']
            if group['task'] in ('structured', 'structured_erased'):
                default_endpoints.append('terminal_control__probe_swapped__swapped_order_sensitive_accuracy')
            endpoints = group['gate_endpoints'] or default_endpoints
        else:
            endpoints = group['gate_endpoints'] or ['late_acquisition', 'late_revision']
        endpoints = [x if x.endswith('_delta_pp') else x+'_delta_pp' for x in endpoints]
        decision = v25_report_gate(stats, endpoints, 100*group['margin'])
        gates.append(dict(group, gate=decision if group['planned_pairing_complete'] else 'inconclusive_incomplete_pairing',
                          descriptive_available_seed_gate=decision,
                          assessed_endpoints=endpoints,
                          interpretation=('Minimal encoder requires retention AND structured/order evidence; '
                                          'a single task margin pass does not establish equivalence.'
                                          if gate_type == 'minimal_encoder' else
                                          'Plasticity margin alone is not a success: assess retention, sample '
                                          'efficiency and resources too. Intervals are descriptive and unadjusted.')))
    return pairs, aggregates, gates, missing


def v25_report_json(path, value):
    path = _V25Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(_v25_json.dumps(v25_report_plain(value), indent=2,
                                        sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def v25_report_table(out, name, rows, compressed=False):
    rows = [v25_report_plain(row) for row in rows]
    columns = list(dict.fromkeys(key for row in rows for key in row))
    path = _V25Path(out) / (name + ('.gz' if compressed else ''))
    opener = _v25_gzip.open if compressed else open
    with opener(path, 'wt', newline='', encoding='utf-8') as handle:
        if columns:
            writer = _v25_csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: v25_report_canonical(v) if isinstance(v, (dict, list)) else v
                                 for k, v in row.items()})
    return dict(file=path.name, rows=len(rows), columns=columns)


def v25_write_outputs(out, cfg, jobs, verification, status):
    """Write compact review products even when some scheduled jobs are unfinished."""
    out = _V25Path(out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = list(jobs)
    summaries, states, failures, development, costs = [], [], [], [], []
    tables = {key: [] for key in ('metrics', 'diagnostics', 'causality', 'controls')}
    stream_metadata = {}
    seen_ids = set()
    for job in jobs:
        meta = v25_report_meta(job)
        if job['id'] in seen_ids:
            failures.append(dict(job_id=job['id'], error='duplicate job ID in manifest'))
            continue
        seen_ids.add(job['id'])
        path = out / 'jobs' / (job['id'] + '.json')
        if not path.exists():
            states.append(dict(meta, status='not_completed'))
            continue
        try:
            result = _v25_json.loads(path.read_text(encoding='utf-8'))
            if not isinstance(result, dict):
                raise ValueError('per-job result must be a JSON object')
            embedded = result.get('job')
            if embedded is not None and (embedded.get('id') != job['id'] or
                    embedded.get('seed') != job.get('seed') or
                    v25_report_configuration(embedded) != v25_report_configuration(job)):
                raise ValueError('saved job configuration differs from manifest')
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            failures.append(dict(meta, error=str(exc)))
            states.append(dict(meta, status='invalid_record'))
            continue
        state = result.get('status', 'completed')
        if result.get('failures'):
            state = 'failed'
        states.append(dict(meta, status=state))
        summaries.extend(v25_report_summaries(job, result))
        for key in tables:
            rows = result.get(key) or []
            if isinstance(rows, dict):
                rows = [rows]
            for row in rows:
                if isinstance(row, dict):
                    # Metadata is authoritative; data cannot silently overwrite it.
                    tables[key].append(dict(row, **meta))
        for failure in result.get('failures') or []:
            failures.append(dict(meta, error=failure))
        costs.append(dict(meta, **v25_report_flatten(result.get('costs') or {}),
                          **v25_report_flatten(result.get('timing') or {}, 'time__')))
        if result.get('development'):
            development.append(dict(meta, development=result['development']))
        stream = result.get('stream_metadata', result.get('stream', result.get('metadata', {})))
        if stream:
            identity = _v25_hashlib.sha256(v25_report_canonical(stream).encode()).hexdigest()[:20]
            entry = stream_metadata.setdefault(identity, dict(metadata=stream, jobs=[]))
            entry['jobs'].append(job['id'])
    aggregates = v25_report_aggregate(summaries)
    pairs, paired_summary, gates, missing_pairs = v25_report_comparisons(jobs, summaries)
    products = []
    for name, rows, compressed in [
            ('SUMMARY_BY_SEED.csv', summaries, False), ('SUMMARY.csv', aggregates, False),
            ('PAIRED_BY_SEED.csv', pairs, False), ('PAIRED_SUMMARY.csv', paired_summary, False),
            ('DECISION_GATES.csv', gates, False), ('MISSING_PAIRS.csv', missing_pairs, False),
            ('JOB_STATUS.csv', states, False), ('FAILURES.csv', failures, False),
            ('COSTS.csv', costs, False), ('DEVELOPMENT.csv', development, False),
            ('METRICS.csv', tables['metrics'], True),
            ('DIAGNOSTICS.csv', tables['diagnostics'], True),
            ('CAUSALITY.csv', tables['causality'], True),
            ('CONTROLS.csv', tables['controls'], True),
            ('CONTROL_SUMMARY.csv', v25_report_diagnostic_aggregate(tables['controls']), True),
            ('CAUSALITY_SUMMARY.csv', v25_report_diagnostic_aggregate(tables['causality']), True),
            ('DIAGNOSTIC_SUMMARY.csv', v25_report_diagnostic_aggregate(tables['diagnostics']), True)]:
        products.append(v25_report_table(out, name, rows, compressed))
    cfg_plain = v25_report_plain(cfg)
    source_path = _V25Path(globals().get('__file__', ''))
    source_hash = _v25_hashlib.sha256(source_path.read_bytes()).hexdigest() if source_path.is_file() else None
    scope = dict(status=status, planned_jobs=len(jobs), unique_planned_job_ids=len(seen_ids),
                 recorded_jobs=sum(s['status'] != 'not_completed' for s in states),
                 completed_jobs=sum(s['status'] in ('completed', 'complete', 'ok') for s in states),
                 unfinished_jobs=sum(s['status'] == 'not_completed' for s in states),
                 failed_or_invalid_jobs=sum(s['status'] in ('failed', 'invalid_record', 'error') for s in states),
                 requested_seeds=sorted({j.get('seed') for j in jobs}, key=str),
                 manifest_complete=(len(states) == len(jobs) and
                                    all(s['status'] in ('completed', 'complete', 'ok') for s in states)),
                 configuration=cfg_plain,
                 interpretation='Only this manifest is assessed; smoke tests and filtered runs do not validate the full study.')
    source = dict(source_file=source_path.name if source_path.is_file() else None,
                  source_sha256=source_hash,
                  configuration_grouping='SHA256 of complete canonical job excluding only ID, seed, and parent/comparison bookkeeping',
                  pairing='Only declared job IDs, matching seeds and explicit read views; no historical roster or inferred controls',
                  confidence_intervals='Descriptive two-sided Student 95% intervals across distinct seeds; n<2 undefined; no multiplicity adjustment',
                  sample_accounting='External examples are per trajectory. Shared sensory branch evaluations and multiple read views are not new external examples. Development is separate.',
                  archive_policy='Compact tabular metrics and metadata only; excludes raw job records, caches, tensors, and checkpoint state.')
    for name, value in [('RUN_STATUS.json', scope), ('MODEL.json', source),
                         ('MANIFEST.json', jobs), ('VERIFICATION.json', verification),
                         ('STREAM_METADATA.json', stream_metadata), ('OUTPUT_INDEX.json', products)]:
        v25_report_json(out / name, value)
    def display(value):
        return 'n/a' if not v25_report_number(value) else '%.2f%%' % (100*value)
    report = [
        '# V25 results', '',
        '%s of %s scheduled trajectories completed. Status: %s.' %
        (scope['completed_jobs'], scope['planned_jobs'], status), '',
        'This report covers the saved manifest. Partial, smoke and filtered runs have limited scientific scope.', '',
        'Source SHA256: `%s`.' % (source_hash or 'unavailable'), '',
        '## Endpoints', '',
        '| Configuration | Task | Read | Seeds | Final | Late acquisition | Late revision |',
        '|---|---|---|---:|---:|---:|---:|']
    for row in aggregates:
        label = '%s / %s / %s [%s]' % (row.get('name'), row.get('encoder_kind'),
                                       row.get('path'), row['config_id'][:8])
        report.append('| %s | %s | %s | %s | %s | %s | %s |' %
                      (label.replace('|', '/'), row['task'], row['view'], row['completed_seeds'],
                       display(row.get('terminal_all_current_mean')), display(row.get('late_acquisition_mean')),
                       display(row.get('late_revision_mean'))))
    report += ['', '## Interpretation and accounting', '',
               '- Every full configuration has its own identity. Numeric precision, protection, routing and circuit overrides cannot merge into an unnamed baseline.',
               '- Late acquisition/revision means the final three corresponding phase endpoints; shorter windows are explicitly flagged. Fixed-rule structured/context tasks have no revision endpoint.',
               '- Learning-curve AUC is the normalized trapezoidal area over the observed checkpoint span, not an estimate of unobserved early performance or online reward.',
               '- Paired intervals use independent seed differences. Single-seed intervals are undefined. Gates are descriptive, unadjusted checks; an inconclusive gate is not a failure.',
               '- The plasticity margin is five percentage points unless the manifest declares otherwise. Passing it alone does not demonstrate better persistent learning.',
               '- The minimal encoder must satisfy its five-point retention and structured/order tolerances together; one passing row is insufficient.',
               '- Hard and blended views share stored supervised state where declared. They are not independent training trajectories and do not multiply sample counts.',
               '- COSTS.csv retains reported encoder, bank, router, protection, static, temporary and evaluator state separately, with branch and local-update counters. Missing measurements remain missing, never zero by assumption.',
               '- Development exposures are separate. Do not add duplicated development records or shared branch evaluations to claim additional unique experience.',
               '- Causal diagnostics hold bank state fixed for address/value interventions; correlations between drift and accuracy are not causal laws.', '',
               'See RUN_STATUS.json for completeness, MANIFEST.json for predeclared comparisons, DECISION_GATES.csv for paired margins, CAUSALITY.csv.gz for the decompositions, and CONTROLS.csv.gz/CONTROL_SUMMARY.csv.gz for explicit order/context interventions.', '']
    (out / 'REPORT.md').write_text('\n'.join(report), encoding='utf-8')
    names = [p['file'] for p in products] + ['RUN_STATUS.json', 'MODEL.json', 'MANIFEST.json',
                'VERIFICATION.json', 'STREAM_METADATA.json', 'OUTPUT_INDEX.json', 'REPORT.md']
    archive = out.parent / (out.name + '_results.zip')
    temporary = archive.with_name(archive.name + '.tmp')
    with _v25_zipfile.ZipFile(temporary, 'w', compression=_v25_zipfile.ZIP_DEFLATED, compresslevel=6) as handle:
        for name in names:
            handle.write(out / name, arcname=name)
    temporary.replace(archive)
    return archive


def v25_reporting_checks():
    import tempfile as _v25_tempfile
    exact = v25_report_interval([1, 2, 3, 4, 5, 6])
    assert exact['n'] == 6 and abs(exact['mean'] - 3.5) < 1e-12
    assert v25_report_interval([.3])['ci95_low'] is None
    a = dict(id='a', seed=1, name='same_name', panel='circuit', memory={'precision': 'float32'})
    b = dict(a, id='b', memory={'precision': 'int8'})
    c = dict(a, id='c', seed=2)
    assert v25_report_meta(a)['config_id'] != v25_report_meta(b)['config_id']
    assert v25_report_meta(a)['config_id'] == v25_report_meta(c)['config_id']
    metrics = []
    for i in range(4):
        metrics.extend([dict(phase='acquire_%s' % i, view='hard', revision=False,
                             external_presentations=2*i+1, all_current=.7, phase_current=.8, prior_current=.6),
                        dict(phase='revision_%s' % i, view='hard', revision=True,
                             external_presentations=2*i+2, all_current=.7, phase_current=.9, prior_current=.6)])
    controls = [dict(control='probe_swapped', view='hard', accuracy=.8,
                     swapped_order_sensitive_accuracy=.75, paired_both_correct=.65,
                     original_target_accuracy=.2, control_information='Both inputs were order-erased.'),
                dict(control='probe_reset', view='hard', accuracy=.5,
                     original_target_accuracy=.5, control_information='Only early context was removed.')]
    result = dict(job=a, status='completed', metrics=metrics, costs={}, failures=[], controls=controls)
    row = v25_report_summaries(a, result)[0]
    assert abs(row['late_acquisition']-.8) < 1e-12 and abs(row['late_revision']-.9) < 1e-12
    assert row['late_acquisition_phases'] == 3 and abs(row['learning_curve_auc']-.7) < 1e-12
    assert not v25_report_delta_is_fraction('terminal_seen_identities')
    assert not v25_report_delta_is_fraction('terminal_current_mse')
    assert not v25_report_delta_is_fraction('terminal_simulation_ticks')
    assert not v25_report_delta_is_fraction('terminal_presentations_to_discovery_among_discovered')
    assert v25_report_delta_is_fraction('terminal_all_current')
    assert v25_report_delta_is_fraction('terminal_replacement_action_discovery_fraction')
    assert v25_report_delta_is_fraction('terminal_control__probe_swapped__paired_both_correct')
    assert row['terminal_control__probe_swapped__swapped_order_sensitive_accuracy'] == .75
    assert row['terminal_control__probe_swapped__paired_both_correct'] == .65
    # Equal normal accuracy is insufficient when transformed order performance
    # loses more than the declared margin. Missing order controls are inconclusive.
    paired_jobs, paired_rows = [], []
    for seed in range(6):
        parent = dict(a, id='parent_%d' % seed, seed=seed, task='structured', name='e3')
        child = dict(parent, id='child_%d' % seed, name='minimal', comparisons=[dict(
            name='minimal', parent_id=parent['id'], gate='minimal_encoder', margin=.05)])
        parent_result = dict(result, job=parent)
        child_controls = [dict(control) for control in controls]
        child_controls[0]['swapped_order_sensitive_accuracy'] = .60
        child_result = dict(result, job=child, controls=child_controls)
        paired_jobs.extend([parent, child])
        paired_rows.extend(v25_report_summaries(parent, parent_result))
        paired_rows.extend(v25_report_summaries(child, child_result))
    _, _, minimum_gates, _ = v25_report_comparisons(paired_jobs, paired_rows)
    assert len(minimum_gates) == 1 and minimum_gates[0]['gate'] == 'fails_margin'
    for paired_row in paired_rows:
        paired_row.pop('terminal_control__probe_swapped__swapped_order_sensitive_accuracy', None)
    _, _, missing_order_gates, _ = v25_report_comparisons(paired_jobs, paired_rows)
    assert missing_order_gates[0]['gate'] == 'inconclusive'
    assert v25_report_gate({'x': dict(n=6, ci95_low=-.04, ci95_high=.02)}, ['x'], .05) == 'passes_margin'
    assert v25_report_gate({'x': dict(n=6, ci95_low=-.06, ci95_high=.02)}, ['x'], .05) == 'inconclusive'
    assert v25_report_gate({'x': dict(n=6, ci95_low=-.08, ci95_high=-.06)}, ['x'], .05) == 'fails_margin'
    with _v25_tempfile.TemporaryDirectory() as folder:
        out = _V25Path(folder) / 'run'
        (out / 'jobs').mkdir(parents=True)
        v25_report_json(out / 'jobs' / 'a.json', result)
        archive = v25_write_outputs(out, {'smoke': True}, [a, b], {'selfcheck': True}, 'interrupted')
        with _v25_zipfile.ZipFile(archive) as handle:
            assert 'REPORT.md' in handle.namelist()
            assert not any(name.startswith('jobs/') for name in handle.namelist())
            control_text = _v25_gzip.decompress(handle.read('CONTROLS.csv.gz')).decode('utf-8')
            control_rows = list(_v25_csv.DictReader(control_text.splitlines()))
            assert len(control_rows) == 2 and control_rows[0]['control_information'] == controls[0]['control_information']
            assert {r['control'] for r in control_rows} == {'probe_swapped', 'probe_reset'}
            summary_text = _v25_gzip.decompress(handle.read('CONTROL_SUMMARY.csv.gz')).decode('utf-8')
            control_summary = list(_v25_csv.DictReader(summary_text.splitlines()))
            assert len(control_summary) == 2 and all(r['seed_count'] == '1' for r in control_summary)
            state = _v25_json.loads(handle.read('RUN_STATUS.json'))
            assert state['completed_jobs'] == 1 and state['unfinished_jobs'] == 1 and not state['manifest_complete']
    return dict(complete_configuration_identity=True, six_seed_student_intervals=True,
                undefined_single_seed_interval=True, phase_endpoints=True,
                gate_tristate=True, interrupted_archive=True, intervention_rows_preserved=True,
                control_descriptors_preserved=True, metric_delta_units=True,
                minimal_encoder_requires_order_parity=True)


# V25 execution, evaluator-only caches, causal diagnostics and recovery.
import atexit
import contextlib
import platform
import signal
import shlex

class V25KeepAwake:
    """Hold macOS idle-sleep assertion only while this Python process exists."""
    def __init__(self,enabled=True): self.enabled=enabled;self.process=None;self.status='not_needed'
    def __enter__(self):
        if self.enabled and sys.platform=='darwin':
            binary='/usr/bin/caffeinate'
            if not Path(binary).is_file():
                self.status='unavailable';print('macOS keep-awake helper unavailable; continuing.',flush=True)
            else:
                try:
                    self.process=subprocess.Popen([binary,'-i','-w',str(os.getpid())],
                        stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
                        start_new_session=True)
                    self.status='active';atexit.register(self.close)
                    print('macOS caffeinate active for this run.',flush=True)
                except OSError as exc:
                    self.status='unavailable';print('Could not start caffeinate: '+str(exc),flush=True)
        elif not self.enabled:self.status='disabled'
        return self
    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:self.process.kill();self.process.wait(timeout=2)
            self.process=None
    def __exit__(self,*args):self.close()

def v25_parse_config(argv=None):
    ap=argparse.ArgumentParser(description='ELM V25: local address/content causal experiment. Automatic macOS caffeinate.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--install-deps',action='store_true')
    ap.add_argument('--out-dir',type=Path)
    ap.add_argument('--resume',type=str,help='Existing run folder, or auto for the latest matching run in this directory.')
    ap.add_argument('--no-caffeinate',action='store_true',help='Disable automatic macOS idle-sleep prevention.')
    ap.add_argument('--seeds',type=int,nargs='+',default=list(V25_SEEDS))
    ap.add_argument('--panels',nargs='+',choices=['all',*V25_PANELS],default=['all'])
    ap.add_argument('--arms',nargs='+',help='Exact configuration names from --list-jobs.')
    ap.add_argument('--max-jobs',type=int,help='Stop after this many new jobs; completed jobs remain resumable.')
    ap.add_argument('--list-jobs',action='store_true')
    ap.add_argument('--self-test',action='store_true')
    ap.add_argument('--smoke',action='store_true',help='Every selected configuration on tiny data, not scientific results.')
    for flag,default in [('dev-cues',64),('dev-passes',6),('group-cues',24),('reverse-cues',12),
                         ('variants',2),('probe-variants',3),('passes',3),('stress-waves',8),('threads',1),('diagnostic-probes',16)]:
        ap.add_argument('--'+flag,type=int,default=default)
    ap.add_argument('--noise',type=float,default=.06)
    ap.add_argument('--learning-rate',type=float,default=.6)
    cfg=ap.parse_args(argv);cfg.neurons=96;cfg.active_fraction=.25;cfg.ridge=.5;cfg.write_energy_floor=.5
    if cfg.smoke:
        cfg.seeds=[17];cfg.dev_cues=16;cfg.dev_passes=1;cfg.group_cues=8;cfg.reverse_cues=4
        cfg.stress_waves=1;cfg.passes=1;cfg.diagnostic_probes=6
    if not cfg.seeds or len(set(cfg.seeds))!=len(cfg.seeds):ap.error('Seeds must be nonempty and distinct.')
    if 'all' in cfg.panels and len(cfg.panels)>1:ap.error('Use all alone or select individual panels.')
    if cfg.group_cues<4 or cfg.group_cues%4:ap.error('group-cues must be a multiple of four, at least four.')
    if cfg.reverse_cues<2 or cfg.reverse_cues%2 or cfg.reverse_cues>2*cfg.group_cues:ap.error('Invalid even reverse-cues count.')
    if min(cfg.dev_cues,cfg.dev_passes,cfg.variants,cfg.probe_variants,cfg.passes,cfg.threads,cfg.diagnostic_probes)<1:ap.error('Counts must be positive.')
    if cfg.stress_waves<0 or not 0<cfg.learning_rate<=1 or not math.isfinite(cfg.noise) or cfg.noise<0:ap.error('Invalid rate/noise/waves.')
    if cfg.max_jobs is not None and cfg.max_jobs<1:ap.error('max-jobs must be positive.')
    if cfg.resume and cfg.out_dir:ap.error('Use either out-dir or resume.')
    return cfg

def v25_pair(features,job):
    path=job['path'];r=np.asarray(features[path[0]])[..., :job['route_dims']]
    z=np.asarray(features[path[1]])
    return r,z

def v25_flat_events(stream):
    events=[];ends=[]
    for p,ph in enumerate(stream['schedule']):
        for cp in ph['checkpoints']:
            for ident,variant,label in cp['events']:events.append((p,int(ident),int(variant),int(label)))
        ends.append(len(events))
    return np.asarray(events,dtype=np.int64),np.asarray(ends,dtype=np.int64)

def v25_cache(out,cfg,job,stream,get_base,source_hash):
    """Never supplied to a learner. Reuses outcome-independent sensory computation."""
    identity=dict(source_sha256=source_hash,seed=job['seed'],kind=job['encoder_kind'],task=job['task'],
        stream=stream.get('full_stream_sha256'),config={k:getattr(cfg,k) for k in
            ('dev_cues','dev_passes','group_cues','reverse_cues','variants','probe_variants','passes','stress_waves','noise')})
    token=sha_bytes(json.dumps(identity,sort_keys=True).encode())[:20]
    directory=out/'feature_cache';directory.mkdir(exist_ok=True)
    file=directory/(token+'.npz');meta_file=directory/(token+'.json')
    if file.exists() and meta_file.exists():
        try:
            meta=json.loads(meta_file.read_text())
            if meta['identity']!=identity:raise ValueError('Cache identity mismatch')
            with np.load(file,allow_pickle=False) as loaded:arrays={k:loaded[k] for k in loaded.files}
            if len(arrays['events'])!=meta['external_presentations']:raise ValueError('Incomplete cache')
            return arrays,meta,False
        except (ValueError,OSError,KeyError,EOFError):
            # Invalid intermediate caches may be regenerated; completed jobs stay intact.
            pass
    started=time.perf_counter()
    base,development=(None,{}) if job['encoder_kind'] in ('minimal','observable') else get_base(job['seed'])
    bundle=V25EncoderBundle(cfg,job['seed'],job['encoder_kind'],base=base)
    events,ends=v25_flat_events(stream)
    codes={};probes={};initial={};N=len(stream['original'])
    learner_seconds=0.;eval_seconds=0.
    # Evaluate in a clone so probe arithmetic/counters never change learner state.
    eval_bundle=bundle.clone();before=eval_bundle.state_digest()
    initial=eval_bundle.features(stream['probe'].reshape(-1,12,32),adapt=False)
    if eval_bundle.state_digest()!=before:raise AssertionError('Probe mutated encoder')
    for key,value in initial.items():
        codes[key]=np.empty((len(events),value.shape[-1]),dtype=np.float64)
        probes[key]=np.empty((len(ends),N*cfg.probe_variants,value.shape[-1]),dtype=np.float64)
    phase_cursor=0;last_notice=time.perf_counter()
    for n,(p,ident,variant,label) in enumerate(events):
        tick=time.perf_counter();f=bundle.features(stream['train'][ident,variant],adapt=True)
        learner_seconds+=time.perf_counter()-tick
        for key in codes:codes[key][n]=np.asarray(f[key]).reshape(-1)
        if n+1==ends[phase_cursor]:
            tick=time.perf_counter();eval_bundle=bundle.clone();before=eval_bundle.state_digest()
            fprobe=eval_bundle.features(stream['probe'].reshape(-1,12,32),adapt=False)
            if before!=eval_bundle.state_digest():raise AssertionError('Impure encoder probes')
            for key in probes:probes[key][phase_cursor]=fprobe[key]
            eval_seconds+=time.perf_counter()-tick;phase_cursor+=1
        if time.perf_counter()-last_notice>=30:
            print('    Sensory trajectory: %d/%d presentations cached.'%(n+1,len(events)),flush=True)
            last_notice=time.perf_counter()
    cost=bundle.costs()
    arrays=dict(events=events,ends=ends)
    for key in codes:
        arrays['train_'+key]=codes[key];arrays['probe_'+key]=probes[key];arrays['initial_'+key]=initial[key]
    # Final control features are encoded once per sensory trajectory and reused.
    control_sources={}
    for group in ('structured_controls','context_controls'):
        for key,value in stream.get(group,{}).items():
            if key.startswith('probe_') and isinstance(value,np.ndarray):control_sources[key]=value
    for control,raw in control_sources.items():
        ctrl=bundle.clone().features(raw.reshape(-1,12,32),adapt=False)
        for key,value in ctrl.items():arrays['control_'+control+'_'+key]=value
    meta=dict(identity=identity,external_presentations=len(events),encoder_costs=cost,
        development=development,bundle_development=getattr(bundle,'development',{}),
        final_encoder_sha256=bundle.state_digest(),learner_seconds=learner_seconds,evaluator_seconds=eval_seconds,
        cache_build_seconds=time.perf_counter()-started,cache_numeric_bytes=sum(a.nbytes for a in arrays.values()),
        frozen_counterfactual_policy='Both sensory branches retained and processed in every dual learner; evaluator reuse does not make branch cost free.')
    tmp=file.with_suffix('.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**arrays)
    tmp.replace(file);v24_atomic_json(meta_file,meta)
    return arrays,meta,True

def v25_cached_pair(cache,job,prefix,index=None):
    result={key:cache[prefix+'_'+key] if index is None else cache[prefix+'_'+key][index] for key in set(job['path'])}
    return v25_pair(result,job)

def v25_prediction(R,Z,centers,W,view='hard'):
    scores=unit_rows(np.atleast_2d(R))@centers.T
    if view=='hard':
        selected=np.argmax(scores,axis=1)
        return np.einsum('nd,ndc->nc',np.atleast_2d(Z),W[selected],optimize=False)
    p=_structural_softmax(8.*scores)
    return np.einsum('nb,nbc->nc',p,np.einsum('nd,bdc->nbc',np.atleast_2d(Z),W,optimize=False),optimize=False)

def v25_causal_rows(job,phase,R0,Z0,R1,Z1,A0,A1,W0,W1,targets,mask,last_write,original_routes,limit):
    indices=np.flatnonzero(mask)
    if not len(indices):return []
    # Fixed deterministic subset; no selection by prediction quality or labels.
    indices=indices[np.linspace(0,len(indices)-1,min(limit,len(indices)),dtype=int)]
    r0,z0,r1,z1=R0[indices],Z0[indices],R1[indices],Z1[indices]
    y=targets[indices];rows=[];loss={};pred={}
    for name,r,z in [('old',r0,z0),('address',r1,z0),('value',r0,z1),('combined',r1,z1)]:
        score=v25_prediction(r,z,A0,W0);pred[name]=score
        loss[name]=float(np.mean(v24_loss(score,y)))
    after_router=v25_prediction(r1,z1,A1,W0);after_writes=v25_prediction(r1,z1,A1,W1)
    old_routes=np.argmax(unit_rows(r0)@A0.T,axis=1);new_routes=np.argmax(unit_rows(r1)@A1.T,axis=1)
    same=new_routes==old_routes
    old_loss=loss['old'];row=dict(phase=phase,probe_count=len(indices),old_mse=old_loss,
        address_mse_change=loss['address']-old_loss,value_mse_change=loss['value']-old_loss,
        combined_mse_change=loss['combined']-old_loss,
        interaction_mse=loss['combined']-loss['address']-loss['value']+old_loss,
        prototype_mse_change=float(np.mean(v24_loss(after_router,y)))-loss['combined'],
        weight_mse_change=float(np.mean(v24_loss(after_writes,y)-v24_loss(after_router,y))),
        total_mse_change=float(np.mean(v24_loss(after_writes,y)))-old_loss,
        phase_route_change_fraction=float(np.mean(new_routes!=old_routes)),
        feature_mse_drift=float(np.mean((z1-z0)**2)),
        encoder_route_change_fraction=float(np.mean(np.argmax(unit_rows(r1)@A0.T,axis=1)!=old_routes)),
        prototype_route_change_fraction=float(np.mean(new_routes!=np.argmax(unit_rows(r1)@A0.T,axis=1))),
        same_route_prediction_displacement=float(np.mean((pred['combined'][same]-pred['old'][same])**2)) if np.any(same) else None,
        same_route_count=int(np.sum(same)),
        latest_write_route_disagreement=float(np.mean(new_routes[last_write[indices]>=0]!=last_write[indices][last_write[indices]>=0]))
            if np.any(last_write[indices]>=0) else None,
        cumulative_route_displacement=float(np.mean(new_routes[original_routes[indices]>=0]!=original_routes[indices][original_routes[indices]>=0]))
            if np.any(original_routes[indices]>=0) else None,
        decomposition_scope='Previous/current phase boundary, same W0 for route/value/prototype reads; all phase writes held for last contrast; ordered interaction, not unique attribution.')
    # Exact telescoping check catches accidental mixed-state diagnostics.
    if not math.isclose(row['combined_mse_change']+row['prototype_mse_change']+row['weight_mse_change'],row['total_mse_change'],abs_tol=1e-10):
        raise AssertionError('Causal decomposition does not telescope')
    rows.append(row);return rows

def v25_run_standard(cfg,job,stream,cache,meta):
    model=V25BankMemory(96,seed_for(job['seed'],'v24_banks'),address_dim=job['route_dims'],**job['memory'])
    metrics=[];diagnostics=[];causality=[];controls=[]
    events=cache['events'];ends=cache['ends'];N=len(stream['original']);P=cfg.probe_variants
    last_route=np.full(N,-1,dtype=int);first_route=np.full(N,-1,dtype=int)
    last_label=np.full(N,-1,dtype=int);n_exposures=np.zeros(N,dtype=int)
    seen_previous=np.zeros(N,dtype=bool);previous_targets=np.asarray(stream['original']).copy()
    previous_R,previous_Z=v25_cached_pair(cache,job,'initial')
    previous_A=model.centers.copy();previous_W=model.V.copy();previous_end=0
    memory_seconds=eval_seconds=0.;all_correct=all_count=0
    views=['hard'] if model.banks==1 else ['hard','blended']
    event_correct=[];first_correct=[];recurrence_correct=[];next_correct=[]
    for phase_index,ph in enumerate(stream['schedule']):
        phase_correct=phase_count=0;collisions=0;changes=[];phase_routes=[]
        previous_route_written=last_route.copy()
        for i in range(previous_end,int(ends[phase_index])):
            _,ident,variant,label=events[i];r,z=v25_cached_pair(cache,job,'train',i)
            started=time.perf_counter();scores=model.output_pair(r,z,'hard')
            hit=int(np.argmax(scores)==label)
            phase_correct+=hit;phase_count+=1;all_correct+=hit;all_count+=1;event_correct.append(hit)
            if n_exposures[ident]==0:first_correct.append(hit)
            else:recurrence_correct.append(hit)
            if n_exposures[ident]==1:next_correct.append(hit)
            info=model.observe_pair(r,z,int(label));memory_seconds+=time.perf_counter()-started
            dest=int(info['selected_bank'])
            other=(last_route==dest)&(np.arange(N)!=ident)&(n_exposures>0)
            collisions+=int(np.any(other));phase_routes.append(dest)
            n_exposures[ident]+=1;last_route[ident]=dest;last_label[ident]=label
            if first_route[ident]<0:first_route[ident]=dest
            changes.append(float(info.get('update_norm',0)))
        previous_end=int(ends[phase_index]);started=time.perf_counter()
        R,Z=v25_cached_pair(cache,job,'probe',phase_index)
        digest=model.state_digest()
        for view in views:
            pred=model.output_pair(R,Z,view)
            metrics.append(dict(phase=ph['name'],revision=bool(ph['revision']),pass_index=cfg.passes,
                external_presentations=previous_end,extra_labeled_presentations=0,total_labeled_presentations=previous_end,
                view=view,prequential_accuracy=all_correct/max(1,all_count) if view=='hard' else None,
                phase_prequential_accuracy=phase_correct/max(1,phase_count) if view=='hard' else None,
                first_exposure_accuracy=float(np.mean(first_correct)) if first_correct and view=='hard' else None,
                recurrence_accuracy=float(np.mean(recurrence_correct)) if recurrence_correct and view=='hard' else None,
                next_recurrence_accuracy=float(np.mean(next_correct)) if next_correct and view=='hard' else None,
                next_recurrence_count=len(next_correct) if view=='hard' else None,
                first_exposure_count=len(first_correct) if view=='hard' else None,
                **v24_metric_scores(pred,ph,stream,cfg)))
        unchanged=seen_previous & (np.asarray(ph['targets'])==previous_targets)
        revised=np.asarray(ph.get('ever_revised',np.zeros(N,dtype=bool)))
        causal_groups=([('old_unchanged',unchanged),('never_revised',unchanged&~revised),
                        ('previously_revised_current',unchanged&revised)] if job['task']=='association' else [])
        for group,eligible in causal_groups:
            for row in v25_causal_rows(job,ph['name'],previous_R,previous_Z,R,Z,previous_A,model.centers,
                previous_W,model.V,np.repeat(ph['targets'],P),np.repeat(eligible,P),
                np.repeat(previous_route_written,P),np.repeat(first_route,P),cfg.diagnostic_probes):
                causality.append(dict(row,probe_group=group))
        counts=np.bincount(phase_routes,minlength=model.banks)
        diagnostic=dict(model.diagnostics());diagnostic.update(phase=ph['name'],external_presentations=previous_end,
            mean_update_norm=float(np.mean(changes)),sampled_write_collision_event_fraction=collisions/max(1,phase_count),
            collision_definition='Current destination shared with at least one other previously written identity (evaluator only).',
            phase_bank_selected_write_counts=counts.tolist(),
            largest_bank_write_fraction=float(np.max(model.counts)/max(1,model.t)))
        diagnostics.append(diagnostic)
        if digest!=model.state_digest():raise AssertionError('Readout evaluation changed learner state')
        previous_R,previous_Z=R,Z;previous_A=model.centers.copy();previous_W=model.V.copy()
        seen_previous=ph['seen'].copy();previous_targets=np.asarray(ph['targets']).copy()
        eval_seconds+=time.perf_counter()-started
    # Correctly intervened structured swaps and paired-context tests.
    ph=stream['schedule'][-1]
    for group in ('structured_controls','context_controls'):
        data=stream.get(group,{})
        for raw_name,target_name in [('probe_swapped','targets_swapped'),('probe_flipped','targets_flipped'),('probe_reset','targets_reset')]:
            if raw_name not in data:continue
            prefix='control_'+raw_name
            if prefix+'_'+job['path'][0] not in cache:continue
            r,z=v25_cached_pair(cache,job,prefix)
            target=np.asarray(data.get(target_name,ph['targets']))
            for view in views:
                pred=model.output_pair(r,z,view)
                controls.append(dict(control=raw_name,view=view,accuracy=accuracy(pred,target,ph['seen'],P),
                    phase=ph['name'],
                    original_target_accuracy=accuracy(pred,ph['targets'],ph['seen'],P),
                    control_information=data.get('description','Explicit transformed probe histories; no learner updates.')))
                row=controls[-1];seen=np.asarray(ph['seen'],dtype=bool)
                base_pred=model.output_pair(R,Z,view)
                normal=base_pred.reshape(N,P,4).argmax(-1);changed=pred.reshape(N,P,4).argmax(-1)
                base_accuracy=accuracy(base_pred,ph['targets'],seen,P)
                if raw_name in ('probe_flipped','probe_swapped'):
                    row.update(normal_accuracy=base_accuracy,paired_both_correct=float(np.mean(
                        (normal[seen]==np.asarray(ph['targets'])[seen,None]) & (changed[seen]==target[seen,None]))))
                if raw_name=='probe_flipped':
                    row.update(balanced_context_accuracy=.5*(base_accuracy+row['accuracy']),
                        changed_prediction_with_identical_late_cue=float(np.mean(normal[seen]!=changed[seen])))
                if raw_name=='probe_reset':
                    alt=accuracy(pred,data['targets_flipped'],seen,P)
                    row.update(reset_balanced_context_accuracy=.5*(row['accuracy']+alt),reset_balanced_context_ceiling=.5)
                if raw_name=='probe_swapped':
                    sensitive=seen & np.asarray(data['order_sensitive_mask'],dtype=bool)
                    row.update(swapped_order_sensitive_accuracy=accuracy(pred,target,sensitive,P),
                        prediction_change_on_sensitive_pairs=float(np.mean(normal[sensitive]!=changed[sensitive])) if sensitive.any() else None)
    costs=dict(model.costs());costs.update(encoder=meta['encoder_costs'],
        external_presentations_completed=model.t,total_learning_presentations=model.t,total_labeled_presentations=model.t,
        extra_labeled_presentations=0,unique_training_identities=int(N),unique_training_observations=int(N*cfg.variants),
        evaluator_cache_numeric_bytes=meta['cache_numeric_bytes'],evaluator_identity_state_bytes=last_route.nbytes*4+n_exposures.nbytes,
        external_sensory_events=model.t*int(stream.get('task_metadata',{}).get('sensory_events_per_episode',2)),
        peak_temporary_numeric_bytes_lower_bound=int(max(R.nbytes+Z.nbytes+model.V.nbytes,previous_W.nbytes*2)),
        temporary_scope='Lower bound for explicit numeric arrays, not process peak RSS.',
        final_memory_sha256=model.state_digest(),final_encoder_sha256=meta['final_encoder_sha256'])
    return dict(status='completed',metrics=metrics,diagnostics=diagnostics,causality=causality,controls=controls,costs=costs,
        failures=[],development=dict(shared_E3=meta['development'],bundle=meta.get('bundle_development',{})),
        timing=dict(memory_loop_seconds=memory_seconds,learner_encoder_seconds=meta['learner_seconds'],
            evaluator_seconds=eval_seconds+meta['evaluator_seconds'],shared_cache_build_seconds=meta['cache_build_seconds']))

def v25_cost_totals(result):
    c=result['costs'];encoder=c.get('encoder',{})
    extra=int(c.get('eligibility_mutable_bytes',0))+int(c.get('optimistic_prior_static_bytes',0))
    c['memory_numeric_bytes_including_teaching']=int(c['persistent_bytes'])+extra
    c['total_learner_numeric_bytes']=c['memory_numeric_bytes_including_teaching']+int(encoder['encoder_array_bytes'])
    c['total_online_mutable_numeric_bytes']=int(c['mutable_bytes'])+int(c.get('eligibility_mutable_bytes',0))+int(encoder['encoder_mutable_array_bytes'])
    c['total_fixed_numeric_bytes']=c['total_learner_numeric_bytes']-c['total_online_mutable_numeric_bytes']
    c['total_byte_scope']='Owned numeric arrays and explicitly counted teaching prior; excludes Python overhead and evaluator caches.'

def v25_runtime_checks():
    # Validate the Mac command and lifetime without starting an OS assertion here.
    from unittest.mock import Mock,patch
    child=Mock();child.poll.return_value=None
    with patch.object(sys,'platform','darwin'),patch.object(Path,'is_file',return_value=True),\
            patch.object(subprocess,'Popen',return_value=child) as launch,patch.object(atexit,'register'):
        with contextlib.redirect_stdout(io.StringIO()),V25KeepAwake() as awake:
            assert awake.status=='active'
            assert launch.call_args.args[0]==['/usr/bin/caffeinate','-i','-w',str(os.getpid())]
        child.terminate.assert_called_once();child.wait.assert_called_once()
    cfg=v25_parse_config(['--smoke']);jobs=v25_jobs(cfg)
    assert len(jobs)==58 and len(set(j['id'] for j in jobs))==58
    assert not any(j['memory']['protection']!='none' for j in jobs if j['teaching'])
    same={j['name']:j for j in jobs}
    assert same['context_reset']['encoder_kind']=='shared_context_reset'
    assert same['context_reset']['task']==same['context_e3']['task']
    for path in ('FF','LL','FL','LF'):
        assert same['dual_'+path+'_base']['encoder_kind']=='dual'
    return dict(caffeinate_command_and_cleanup_mocked=True,caffeinate_actual_macos_execution_tested=False,
        fixed_roster_58=True,context_reset_same_input_stream=True,dual_state_cost_matching=True)

def v25_checks():
    tiny=v25_parse_config(['--smoke'])
    base,_=v24_develop_encoder(tiny,17,'E3')
    return dict(memory=v25_memory_checks(),encoder=v25_encoder_checks(tiny,base),
        tasks=v25_task_checks(tiny),teaching=v25_teaching_checks(),reporting=v25_reporting_checks(),runtime=v25_runtime_checks(),
        scope='Deterministic semantic and arithmetic checks; not scientific outcomes.',
        numpy_version=np.__version__,torch_version=torch.__version__,python_version=sys.version,
        platform=platform.platform())

def v25_config_identity(cfg,jobs,source_hash):
    ignored={'out_dir','resume','install_deps','max_jobs','list_jobs','self_test','no_caffeinate'}
    values={k:v for k,v in v25_report_plain(vars(cfg)).items() if k not in ignored}
    return dict(source_sha256=source_hash,configuration=values,jobs=jobs)

def v25_resume_config(cfg,argv,source_hash):
    if not cfg.resume:return None
    if cfg.resume=='auto':
        candidates=[]
        for path in Path.cwd().glob('elm_fly_v25_*/RUN_IDENTITY.json'):
            try:
                saved=json.loads(path.read_text())
                if saved['source_sha256']==source_hash:candidates.append(path)
            except (OSError,ValueError,KeyError):continue
        if not candidates:raise SystemExit('No matching V25 run found. Give --resume FOLDER.')
        folder=max(candidates,key=lambda p:p.stat().st_mtime).parent
    else:folder=Path(cfg.resume).expanduser().resolve()
    if not (folder/'RUN_IDENTITY.json').is_file():raise SystemExit('Missing run identity in '+str(folder))
    saved=json.loads((folder/'RUN_IDENTITY.json').read_text())
    if saved.get('source_sha256')!=source_hash:raise SystemExit('Resume requires the original unmodified script. Source hash differs.')
    old=json.loads((folder/'CONFIG.json').read_text())
    explicit={arg.split('=',1)[0][2:].replace('-','_') for arg in argv if arg.startswith('--')}
    runtime={'out_dir','resume','install_deps','max_jobs','list_jobs','self_test','no_caffeinate'}
    for key,value in old.items():
        if key in runtime:continue
        if key in explicit and v25_report_plain(getattr(cfg,key))!=value:
            raise SystemExit('Resume argument differs from original configuration: '+key)
        setattr(cfg,key,value)
    return folder

def v25_main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    cfg=v25_parse_config(argv)
    source_hash=sha_bytes(Path(__file__).read_bytes())
    resume_folder=v25_resume_config(cfg,argv,source_hash)
    jobs=v25_jobs(cfg)
    if not jobs:raise SystemExit('No matching configurations. Use --list-jobs.')
    if cfg.list_jobs:
        print('%d trajectories; %d seeds; %d configurations per seed.'%(len(jobs),len(cfg.seeds),len(jobs)//len(cfg.seeds)))
        for j in jobs:
            if j['seed']==cfg.seeds[0]:print('%-12s %-34s %-22s %s'%(j['panel'],j['name'],j['encoder_kind'],j['path']))
        return 0
    with V25KeepAwake(not cfg.no_caffeinate) as awake:
        dependencies(cfg.install_deps);torch.set_num_threads(cfg.threads)
        print('Checking local-learning interfaces, memory arithmetic and reporting.',flush=True)
        verification=v25_checks();verification['keep_awake_status']=awake.status
        if cfg.self_test:
            print(json.dumps(v25_report_plain(verification),indent=2));return 0
        identity=v25_config_identity(cfg,jobs,source_hash)
        fingerprint=sha_bytes(json.dumps(identity,sort_keys=True).encode())
        out=resume_folder or (cfg.out_dir or Path('elm_fly_v25_'+time.strftime('%Y%m%d_%H%M%S')+'_'+str(os.getpid()))).expanduser().resolve()
        if resume_folder:
            prior=json.loads((out/'RUN_IDENTITY.json').read_text())
            if prior.get('fingerprint')!=fingerprint:raise SystemExit('Resume experiment manifest/configuration differs.')
        else:out.mkdir(parents=True,exist_ok=False)
        (out/'jobs').mkdir(exist_ok=True)
        v25_report_json(out/'RUN_IDENTITY.json',dict(fingerprint=fingerprint,identity=identity,source_sha256=source_hash,version=V25_VERSION))
        v25_report_json(out/'CONFIG.json',vars(cfg))
        v25_report_json(out/'DESIGN_POLICY.json',v25_design_policy())
        verification['design_policy']=v25_design_policy()
        verification['development_only_local']=True
        print('Output: '+str(out),flush=True)
        print('%d predeclared trajectories. Completed jobs resume individually.'%len(jobs),flush=True)
        if cfg.smoke:print('SMOKE EXECUTION CHECK: reduced data, not study results.',flush=True)
        encoders={};streams={};caches={};completed=performed=failures=0;active_seed=None
        status='completed';last_archive=time.perf_counter()
        def get_base(seed):
            if seed not in encoders:
                print('Developing sensory-only E3 encoder, seed %s.'%seed,flush=True)
                encoders[seed]=v24_develop_encoder(cfg,seed,'E3')
            return encoders[seed]
        old_handler=signal.getsignal(signal.SIGTERM)
        def terminate_run(signum,frame):raise KeyboardInterrupt('Termination requested')
        signal.signal(signal.SIGTERM,terminate_run)
        try:
            for position,job in enumerate(jobs):
                path=out/'jobs'/(job['id']+'.json')
                if path.is_file():
                    saved=json.loads(path.read_text())
                    if saved.get('job')!=job or saved.get('source_sha256')!=source_hash:
                        raise ValueError('Saved job identity differs: '+job['id'])
                    if saved.get('status')=='completed' and not saved.get('failures'):
                        completed+=1;continue
                if cfg.max_jobs is not None and performed>=cfg.max_jobs:
                    status='stopped_at_job_limit';break
                if active_seed!=job['seed']:
                    encoders.clear();streams.clear();caches.clear();active_seed=job['seed']
                tick=time.perf_counter()
                try:
                    task=job['task']
                    if task not in streams:streams[task]=v25_prepare_stream(cfg,job['seed'],task)
                    stream=streams[task]
                    if job['teaching']:
                        base,development=get_base(job['seed'])
                        bundle=V25EncoderBundle(cfg,job['seed'],job['encoder_kind'],base=base)
                        make_model=lambda:V25BankMemory(96,seed_for(job['seed'],'v24_banks'),address_dim=job['route_dims'],**job['memory'])
                        result=v25_run_teaching(cfg,job,bundle,stream,make_model)
                        result['development']=dict(shared_E3=development,bundle=bundle.development)
                    else:
                        key=(job['encoder_kind'],task)
                        if key not in caches:
                            print('  Encoding %s / %s.'%key,flush=True)
                            arrays,meta,built=v25_cache(out,cfg,job,stream,get_base,source_hash)
                            caches[key]=(arrays,meta)
                        result=v25_run_standard(cfg,job,stream,*caches[key])
                    v25_cost_totals(result)
                    result.update(job=job,source_sha256=source_hash,
                        stream_metadata=dict(task=stream.get('task_metadata',{}),stream_sha256=v25_stream_hash(stream),
                            encoder_context_reset_bin=5 if job['encoder_kind']=='shared_context_reset' else None,
                            inherited_stream_sha256=stream.get('full_stream_sha256'),
                            schedule=[dict(name=p['name'],revision=p['revision'],seen=int(p['seen'].sum())) for p in stream['schedule']]))
                    result['timing']['actual_job_wall_seconds']=time.perf_counter()-tick
                    try:
                        import resource
                        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                        result['costs']['process_peak_rss_through_job_bytes']=int(rss*(1 if sys.platform=='darwin' else 1024))
                        result['costs']['process_rss_scope']='Process lifetime high-water mark including shared caches, evaluator, dependencies and prior jobs; not per-arm temporary memory.'
                    except ImportError:pass
                    # Disallow nonfinite results before declaring a completed job.
                    json.dumps(v25_report_plain(result),allow_nan=False)
                except Exception as exc:
                    import traceback
                    failures+=1;result=dict(job=job,source_sha256=source_hash,status='failed',
                        failures=[str(exc)],traceback=traceback.format_exc(),metrics=[],costs={})
                    print('FAILED '+job['name']+': '+str(exc),flush=True)
                v25_report_json(path,result);performed+=1
                if result['status']=='completed':completed+=1
                print('[%d/%d] seed %s %s: %s (%.1fs)'%(position+1,len(jobs),job['seed'],job['name'],result['status'],time.perf_counter()-tick),flush=True)
                if performed%6==0 or time.perf_counter()-last_archive>60:
                    v25_write_outputs(out,cfg,jobs,verification,'running');last_archive=time.perf_counter()
        except KeyboardInterrupt:
            status='interrupted';print('Stopping; completed jobs and a partial ZIP will be preserved.',flush=True)
        finally:signal.signal(signal.SIGTERM,old_handler)
        if status=='completed' and failures:status='completed_with_failures'
        archive=v25_write_outputs(out,cfg,jobs,verification,status)
        print('Results ZIP: '+str(archive),flush=True)
        print('Status: %s; %d/%d completed.'%(status,completed,len(jobs)),flush=True)
        print('Resume: python3 %s --resume %s'%(shlex.quote(Path(__file__).name),shlex.quote(str(out))),flush=True)
        return 0 if status=='completed' else 2


if __name__=='__main__':
    raise SystemExit(v25_main())
