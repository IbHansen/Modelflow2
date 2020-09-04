// 2019.06.24 10:37 

// ************************************************
// *                 SMEC F19                     *
// ************************************************

// ÆNDRINGER I FORHOLD TIL E18:

// A.   Bittesmå fejlrettelser i formelfilbidder /11.01.19 DG
// B.   Vækstkorrektionsled indført i PYF-ligninger for B,I,T /25.02.19 DG
// C.   Beregning af Jled/kfaktor ved eksogenisering/alternativ ligning af IOFU, SSYEJ, SIQEJ, IVOS /25.02.19 DG
// D.   IXOS = FIBOS*PIBOS+FIMXOS*PIMXOS indført som ligning /25.02.19 DG
// E.   TYOP<I,Y>D har ændret navn til TYOP<I,Y>X som andre eksogene udgangsskøn (gange fx satsregulering) /25.02.19 DG
// F.   TYRHS skiftet til satsregulering i stedet for prisregulering /25.02.19 DG
// G.   PIOFU: Ændret til samme formulering som andre IO-bestemte priser, dvs. PIOFU = PXO*KPIOFU /26.02.19 DG
// H.   PNCP: KP-led fjernet og nettoprisen er i stedet defineret ud fra forbrugsudgift beregnet med nettopriserne /26.02.19 DG
// I.   PTTYP: Eksogeniseringsmulighed tilføjet (som for PSRTY) /26.02.19 DG
// J.   FCH: Niveaukorrektion (KFCHW) tilføjet (=0 i estimationen) /26.02.19 DG
// K.   BTYD udgår, da det blot var en tabelvariabel vi ikke længere forholdt os til /26.02.19 DG
// L.   TAPOK (kirkeskat): TTAPOK -> TKS (da det er en skattesats (samme navn i ADAM))
//                         KTAPOK -> BTAPOK (da det er den andel af indkomsten, der er medlem af folkekirken, ikke bare en tilfældig korrektionsfaktor)
//                         /11.03.19 DG
// M.   PCAHJ og CAHJ (=PCA,CA) er afskaffet, men desværre er FCAHJ stadig nødvendig for at sikre konvergens /12.03.19 DG
// N.   Overflødige "langsigtsstørrelser" er fjernet (vi startede med både kvoter og niveauer, har kun beholdt dem der bruges i fejlkorr.) /12.03.19 DG
// O.   TPPILFTY indført som nye pensionsindbetalinger for indkomstoverførselsmodtagere /19.04.19 DG
// P.   Off.forbrug ekskl. afskrivninger (COZ) er tilføjet som "tabelvariabel" /01.05.19 DG
// Q.   Rettet ligning for FIVOS + tilføjet KFIVOS i AFTER (gjort direkte i SMEC.FRM til F19) /24.06.19 DG
// Samt diverse (forhåbentlig uskyldige og gavnlige...) oprydning i formelfilerne /DG løbende mellem E18 og F19...

// -----------------------------------------------------------------

// SENESTE REESTIMATION: F18
// Se m:\MODEL\Smec\Arkiv\SMEC_f18\Estimation\

// -----------------------------------------------------------------

// INDHOLDSFORTEGNELSE
// 1.  Eksport, mængder
// 2.  Privat forbrug, mængder
//     2.1 Forbrugsfunktion og samlet forbrug
//     2.2 Forbrugsallokeringssystem
//     2.3 Bilkøb
//     2.4 Forbrug af boligbenyttelse
// 3.  Investeringer, kapitalapparat og afskrivninger
//     3.1 erhvervsfordelte investeringer og kapitalapparat
//     3.2 aggregater og afskrivninger
// 4.  Boliginvesteringer og kontantpris
// 5.  Import, mængder
//     5.1 Import af varer ekskl. energi
//     5.2 Import af tjenester ekskl. søfart og turisme
//     5.3 Import af energi, turisme, søfart
//     5.4 Aggregater
// 6.  IO-koefficienter
//     6.1 Importsubstitution, industrivarer
//     6.2 Importsubstitution, tjenester
//     6.3 IO-justeringer
//     6.4 Særbehandlede IO-koefficienter
// 7.  BVT mv., mængder
//     7.1 Produktionsværdier
//     7.2 BVT
//     7.3 Råvarekøb
//     7.4 Aggregater
// 8.  Beskæftigelse, produktivitet, arbejdsstyrke og ledighed
//     8.1 Beskæftigelse i timer
//     8.2 Beskæftigelse i personer
//     8.3 Arbejstid
//     8.4 Timeproduktivitet
//     8.5 Arbejdsstyrke, ledige og aktiverede
// 9.  Den offentlige sektor
//     9.1 Offentlig produktion, forbrug og investeringer
//     9.2 Nettofordringserhvervelse og rentestrømme
//     9.3 Offentlige transfereringer mv.
//     9.4 Samlede og direkte skatter
//     9.5 Indirekte skatter, subsidier og andre skatter samt øvrige indtægter
// 10. Betalingsbalance og udlandsgæld
// 11. Privat sektors nettofordringserhvervelse, indkomst og formue
//     11.1 indkomster
//     11.2 forbrugsbestemmende formue og nettofordringserhvervelse
//     11.3 pensioner
// 12. Løn
//     12.1 Løn
//     12.2 Lønsummer og lønkvoter
//     12.3 Implicit timeløn
// 13. Deflatorer
//     13.1 BVT-deflatorer
//     13.2 Produktionsværdi-deflatorer
//     13.3 Netto-priser på endelig anvendelse
//     13.4 Markeds-priser på endelig anvendelse mv.
//     13.5 Energipriser i udenrigshandlen
//     13.6 Deflatorer på aggregater
// 14. Værdier (løbende priser)
//     14.1 Tilgang
//     14.2 BNP, BVT, forbrug, investeringer, import og eksport
// 15. Renter og valutakurs
// 16. Strukturelle niveauer og gaps
                                                     
// ************************************************  
// * 1.  EKSPORT, MÆNGDER                         *  
// ************************************************  
// Eksport af industrivarer (ikke-energi-varer)
FRML _DJRD     Log(peeiw)    = 0.22733*Log(((udliht/udvyfh)/effkr)*100) + 0.77267*Log(pmi) + kpeeiw ;
FRML _SJRD     Dlog(peei)    = 0.48415*Dlog(((udliht/udvyfh)/effkr)) + 0.42319*Dlog(pmi) + 0.09265*Dlog(pme) + gpeei
                              -0.30834*Log(peei[-1]/peeiw[-1]) ;

FRML _DJRD     Log(feeiw)    = 1.96443*Log(udfy_s) + kfeeiw ;
FRML _SJRD     Dlog(feei)    = 3.05178*Dlog(udfy) + gfeei
                               -0.12702*Log(feei[-1]/feeiw[-1]);

FRML _DJRD     Log(bfeiw)    = -2.5*Log(pei/peei) + kbfeiw;
FRML _SJRD     Dlog(fei)     = 0.59085*Dlog(feei) - 0.56055*Dlog(pei/peei) + gfei
                               -0.2*(Log(fei[-1]/feei[-1])-Log(bfeiw[-1]));

// Eksport af energi
FRML _GJRD     fee           = fee[-1]*(fxn/fxn[-1]) + Jfee ;

// Eksport af øvrig tjenesteeksport
FRML _GJRD     peesq         = peesq[-1]*(1+0.80*Dlog((udliht/udvyfh)/effkr)+0.20*Dlog(pmsq)) ;
FRML _DJRD     Log(fesqw)    = 1.85716*Log(udfy_s) - 0.80825*Log(pesq/peesq)
                               -0.25715*d4803  + kfesqw ;
FRML _SJRD     Dlog(fesq)    = 3.66554*(udfy/udfy_s-1) - 1.48551*Dlog(pesq/peesq) + gfesq
                               -0.53855*Log(fesq[-1]/fesqw[-1]) ;

// Eksport af søfart
FRML _DJRD     Log(fessw)    = Log(udfy_s) + (0.01543+0.09685*d8599)*dtfess
                               - 3.16194*d8599 + kfessw;
FRML _SJRD     Dlog(fess)    = 2.75878*(udfy/udfy_s-1) + gfess
                               -0.26918*Log(fess[-1]/fessw[-1]);

// Eksport af turisme
FRML _DJRD     Log(bfetw)    = kbfetw;
FRML _SJRD     Dlog(fet)     = 1.33127*Dlog(udfy) - 0.92460*(.5*Dlog(pet/pmi)+.5*Dlog(pet[-1]/pmi[-1]))  + gfet
                               -0.21769*(Log(fet[-1]/udfy_s[-1])-Log(bfetw[-1])) ;

// AGGREGATER
FRML _I        fev           = (fee*pee[-1]+fei*pei[-1])/pev[-1] ;
FRML _I        fes           = (fess*pess[-1]+fesq*pesq[-1])/pes[-1] ;
FRML _I        fest          = (fet*pet[-1]+fes*pes[-1])/pest[-1] ;
FRML _I        fe            = (fev*pev[-1]+fest*pest[-1])/pe[-1] ;

                                                     
// ************************************************  
// * 2.  PRIVAT FORBRUG                           *  
// ************************************************  
// 2.1 SAMLET FORBRUG
// ******************

// SAMLET PRIVAT FORBRUG
FRML _DJRD     Log(bfcpw)    = 0.12688*Log((Wcp/pcp)/fYdl) - 1.42316*bu5064 + kbfcpw ;
FRML _SJRD     Dlog(fCp)     = 0.34875*Dlog(fYdk) + 0.25 *Dlog(q) + gfcp
                                -0.30668*(Log(fCp[-1]/fYdl[-1])-Log(bfcpw[-1])) ;

FRML _D        BCPK          =  FCP/FYDK ;
FRML _D        BCPL          =  FCP/FYDL ;

// 2.2 FORBRUGSALLOKERINGSSYSTEM
FRML _I        FCAHJ         = (FCDLU*PCDLU(-1)+FCH*PCH(-1)-FET*PET(-1))/PCA(-1) ;
FRML _D        PCA           = CA/FCAHJ ;
FRML _D        FCA           = (FCP*PCP[-1]-FCB*PCB[-1])/PCA[-1] ;

FRML _I        FCDLU         = (FCE*PCE(-1)+FCG*PCG(-1)+FCV*PCV(-1)+FCS*PCS(-1)+FCT*PCT(-1))/PCDLU(-1) ;
FRML _I        PCDLU         = (FCE*PCE+FCG*PCG+FCV*PCV+FCS*PCS+FCT*PCT)/FCDLU ;
FRML _I        CDLU          = FCDLU*PCDLU ;

FRML _SJ_      FCE           =  2543.63 +       0*DTFC + .886207*(FCE(-1)-0.00*FET(-1)*PET(-1)/PCE(-1)) + .010897*CQO/PCE + 0.00*FET*PET/PCE ;
FRML _SJ_      FCG           =  1070.66 +       0*DTFC + .846404*(FCG(-1)-0.05*FET(-1)*PET(-1)/PCG(-1)) + .007898*CQO/PCG + 0.05*FET*PET/PCG ;
FRML _SJ_      FCV           =  37716.1 - 2015.55*DTFC + .299723*(FCV(-1)-0.50*FET(-1)*PET(-1)/PCV(-1)) + .558405*CQO/PCV + 0.50*FET*PET/PCV ;
FRML _SJ_      FCS           = -45384.8 +       0*DTFC + .749489*(FCS(-1)-0.45*FET(-1)*PET(-1)/PCS(-1)) + .384341*CQO/PCS + 0.45*FET*PET/PCS ;
FRML _SJ_      FCT           = -5717.84 +       0*DTFC + .845058*(FCT(-1)-0.00*FET(-1)*PET(-1)/PCT(-1)) + .038459*CQO/PCT + 0.00*FET*PET/PCT ;

FRML _D        CQ            = CP - PCB*FCB - PCH*FCH ;
FRML _S        CQO           = CQ - ( PCE*(( 2543.63 +       0*DTFC + .886207*(FCE(-1)-0.00*FET(-1)*PET(-1)/PCE(-1))) + JFCE)
                                     +PCG*(( 1070.66 +       0*DTFC + .846404*(FCG(-1)-0.05*FET(-1)*PET(-1)/PCG(-1))) + JFCG)
                                     +PCV*(( 37716.1 - 2015.55*DTFC + .299723*(FCV(-1)-0.50*FET(-1)*PET(-1)/PCV(-1))) + JFCV)
                                     +PCS*((-45384.8 +       0*DTFC + .749489*(FCS(-1)-0.45*FET(-1)*PET(-1)/PCS(-1))) + JFCS)
                                     +PCT*((-5717.84 +       0*DTFC + .845058*(FCT(-1)-0.00*FET(-1)*PET(-1)/PCT(-1))) + JFCT) ) ;


// 2.3 BILKØB
FRML _DJRD     Log(bfcbw)    = -0.34917*Log(pcb/pcp) + kbfcbw;
FRML _SJRD     Dlog(fcb)     = 2.27727*Dlog(fydl) - 0.74251*Dlog(pcb/pcp) + 1.60549*Dlog(phk/pcp) + gfcb
                               -0.41422*(Log(fcb[-1]/fydl[-1])-Log(bfcbw[-1]));

FRML _D        fkcb          = fcb + (1-dprcb)*fkcb[-1] ;

// 2.4 BOLIGFORBRUG
FRML _SJRD     Log(fch)      = 0.82443*Log(fkbh) + kfchw ;

                                                     
// ************************************************  
// * 3.  INVESTERINGER OG KAPITALAPPARAT          *  
// ************************************************  
// 3.1 ERHVERVSFORDELTE INVESTERINGER OG KAPITALAPPARAT

// INFLATIONSFORVENTNINGER
FRML _D        gpib          = (pib - pib[-1])/pib[-1] ;
FRML _D        gpibe         = (1-Dgpibe)*( 0.25*gpib+0.75*gpibe[-1] + Jgpibe )
                                + Dgpibe*Zgpibe ;

FRML _D        gpim          = (pim - pim[-1])/pim[-1] ;
FRML _D        gpime         = (1-Dgpime)*( 0.25*gpim+0.75*gpime[-1] + Jgpime )
                                + Dgpime*Zgpime ;

// I-ERHVERV
FRML _D        PKMI          = ((1-TSDS*BIVM)/(1-TSDS))*(0.5*IWLO*(1-TSDS)+0.5*IWLO+DPRMI-GPIME+RPI)*PIM ;
FRML _D        PKBI          = ((1-TSDS*BIVB)/(1-TSDS))*(0.5*IWLO*(1-TSDS)+0.5*IWLO+DPRBI-GPIBE+RPI)*PIB ;
FRML _D        pkzi          = (pkmi*fKmi + pkbi*fKbi)/fKzi ;

FRML _DJRD     Log(fKziw)    =   dtalfai*Log((1-dtalfai)/dtalfai)
                               + Log(fyfi) - Log(dtai)
                               + dtalfai*(Log(li)-Log(pkzi));
FRML _D        fKbiw         = bkbiw*fkziw;
FRML _D        fKmiw         = (1-bkbiw)*fkziw;

FRML _SJRD     Dlog(fkmi)    =   0.21343*(0.5*Dlog(udfy)+0.5*Dlog(udfy[-1]))
                               + 0.47991*Dlog(fkmi[-1])
                               + 0.00428*Dlog(fkmiw)
                               + 0.00684
                               - 0.06901*Log(fkmi[-1]/fkmiw[-1]);
FRML _D        FIMI          = (Diff(FKMI) + DPRMI*FKMI[-1])*KFKMI ;

FRML _SJRD     Dlog(fkbi)    =   0.12431*(0.5*Dlog(fyfi)+0.5*Dlog(fyfi[-1]))
                               + 0.55323*Dlog(fkbi[-1])
                               + 0.00615
                               - 0.05407*Log(fkbi[-1]/fkbiw[-1]);
FRML _D        FIBI          = (Diff(FKBI) + DPRBI*FKBI[-1])*KFKBI ;

FRML _D        fKzi          = fKmi + fKbi;
FRML _D        bkbi          = fKbi/fKzi ;

FRML _G        Dlog(DTAI)    = VTFPI/100 + (Log(FKZI)-Log(HQI))*Diff(DTALFAI) ;
FRML _G        DTAIF         = FYFI/( (HQI**DTALFAI) * (FKZI**(1-DTALFAI)) ) ;
FRML _G        VTFPIF        = 100*(Dlog(DTAIF)-((Log(FKZI)-Log(HQI))*Diff(DTALFAI))) ;
FRML _G        DTALFAIF      = LI*HQI/(LI*HQI+PKMI*FKMI+PKBI*FKBI) ;

// T-ERHVERV
FRML _D        PKMT          = ((1-TSDS*BIVM)/(1-TSDS))*(0.5*IWLO*(1-TSDS)+0.5*IWLO+DPRMT-GPIME+RPT)*PIM ;
FRML _D        PKBT          = ((1-TSDS*BIVB)/(1-TSDS))*(0.5*IWLO*(1-TSDS)+0.5*IWLO+DPRBT-GPIBE+RPT)*PIB ;
FRML _D        pkzt          = (pkmt*fkmt + pkbt*fkbt)/fkzt;

FRML _DJRD     Log(fkztw)    =   dtalfat*Log((1-dtalfat)/dtalfat)
                               + Log(fyft) - Log(dtat)
                               + dtalfat*(Log(lt)-Log(pkzt));
FRML _D        fkbtw         = bkbtw*fkztw;
FRML _D        fkmtw         = (1-bkbtw)*fkztw;

FRML _SJRD     Dlog(fkmt)    =   0.36598*Dlog(fyft)
                               + 0.63754*Dlog(fkmt[-1])
                               + 0.00813
                               - 0.09182*Log(fkmt[-1]/fkmtw[-1]);
FRML _D        FIMT          = (Diff(FKMT) + DPRMT*FKMT[-1])*KFKMT ;

FRML _SJRD     Dlog(fkbt)    =   0.08299*Dlog(fyft)
                                -0.00619*Dlog(fyft[-1])
                               + 0.79133*Dlog(fkbt[-1])
                               + 0.00137
                               - 0.07*Log(fkbt[-1]/fkbtw[-1]);
FRML _D        FIBT          = (Diff(FKBT) + DPRBT*FKBT[-1])*KFKBT ;

FRML _D        fKzt          = fKmt + fKbt;
FRML _D        bkbt          = fKbt/fKzt ;

FRML _G        Dlog(DTAT)    = VTFPT/100 + (Log(FKZT)-Log(HQT))*Diff(DTALFAT) ;
FRML _G        DTATF         = FYFT/((HQT**DTALFAT) * (FKZT**(1-DTALFAT))) ;
FRML _G        VTFPTF        = 100*(Dlog(DTATF)-((Log(FKZT)-Log(HQT))*Diff(DTALFAT))) ;
FRML _G        DTALFATF      = LT*HQT/(LT*HQT+PKMT*FKMT+PKBT*FKBT) ;

// B-ERHVERV
FRML _D        PKMB          = ((1-TSDS*BIVM)/(1-TSDS))*(0.5*IWLO*(1-TSDS)+0.5*IWLO+DPRMB-GPIME+RPB)*PIM ;
FRML _D        PKBB          = ((1-TSDS*BIVB)/(1-TSDS))*(0.5*IWLO*(1-TSDS)+0.5*IWLO+DPRBB-GPIBE+RPB)*PIB ;
FRML _D        pkzb          = (pkmb*fkmb + pkbb*fkbb)/fkzb;
FRML _DJRD     Log(fKzbw)    = dtalfab*Log((1-dtalfab)/dtalfab)
                               + Log(fyfb) - Log(dtab)
                               + dtalfab*(Log(lb)-Log(pkzb));
FRML _D        fKbbw         = bkbbw*fkzbw;
FRML _D        fKmbw         = (1-bkbbw)*fkzbw;
FRML _SJRD     Dlog(fkzb)    =   0.08695*Dlog(fyfb)
                               + 0.49456*Dlog(fkzb[-1])
                               + 0.00537
                               - 0.01932*Log(fkzb[-1]/fkzbw[-1]);

FRML _D        FIMB          = (Diff(FKMB) + DPRMB*FKMB[-1])*KFKMB ;
FRML _GJRD     FKBB          = FKZB*BKBBW ;
FRML _GJRD     FKMB          = FKZB*(1-BKBBW) ;
FRML _D        FIBB          = (Diff(FKBB) + DPRBB*FKBB[-1])*KFKBB ;
FRML _D        bkbb          = fKbb/fKzb ;

FRML _G        Dlog(DTAB)    = VTFPB/100 + (Log(FKZB)-Log(HQB))*Diff(DTALFAB) ;
FRML _G        DTABF         = FYFB/( (HQB**DTALFAB) * (FKZB**(1-DTALFAB)) ) ;
FRML _G        VTFPBF        = 100*(Dlog(DTABF)-((Log(FKZB)-Log(HQB))*Diff(DTALFAB))) ;
FRML _G        DTALFABF      = LB*HQB/(LB*HQB+PKMB*FKMB+PKBB*FKBB) ;


// A-ERHVERV
FRML _SJRD     Dlog(fima)    = 0.14574*(Dlog(pyfa/pyf)
                                  -(1/5*(Dlog(pyfa[-1]/pyf[-1]) + Dlog(pyfa[-2]/pyf[-2])
                                       + Dlog(pyfa[-3]/pyf[-3]) + Dlog(pyfa[-4]/pyf[-4])
                                       + Dlog(pyfa[-5]/pyf[-5]))))
                               -2.90777*Diff(iwbz[-1])
                               + Dlog(fyfa);
FRML _D        FKMA          = FKMA[-1] + FIMA/KFKMA - DPRMA*FKMA[-1] ;

FRML _SJRD     Dlog(fiba)    = 0.75*Dlog(fima) + 0.25*Dlog(fima[-1]);
FRML _D        FKBA          = FKBA[-1] + FIBA/KFKBA - DPRBA*FKBA[-1] ;

// E-ERHVERV
FRML _GJRD     FIME          = KFIME*FYFE ;
FRML _D        FKME          = FKME[-1] + FIME/KFKME - DPRME*FKME[-1] ;
FRML _GJRD     FIBE          = KFIBE*FYFE ;
FRML _D        FKBE          = FKBE[-1] + FIBE/KFKBE - DPRBE*FKBE[-1] ;

// N-ERHVERV
FRML _GJRD     FIMN          = KFIMN*FYFN ;
FRML _D        FKMN          = FKMN[-1] + FIMN/KFKMN - DPRMN*FKMN[-1] ;
FRML _GJRD     FIBN          = KFIBN*FYFN ;
FRML _D        FKBN          = FKBN[-1] + FIBN/KFKBN - DPRBN*FKBN[-1] ;

// S-ERHVERV
FRML _GJRD     FIMS          = KFIMS*FYFS ;
FRML _D        FKMS          = FKMS[-1] + FIMS/KFKMS - DPRMS*FKMS[-1] ;
FRML _GJRD     FIBS          = KFIBS*FYFS ;
FRML _D        FKBS          = FKBS[-1] + FIBS/KFKBS - DPRBS*FKBS[-1] ;

// O-ERHVERV
FRML _G        FIMXO         = KFIMO*FIMXOS ;
FRML _G        FIMO          = KFIMO*FIMOS ;
FRML _D        FKMO          = FKMO[-1] + FIMO/KFKMO - DPRMO*FKMO[-1] ;
FRML _G        FIBO          = KFIBO*FIBOS ;
FRML _D        FKBO          = FKBO[-1] + FIBO/KFKBO - DPRBO*FKBO[-1] ;


// 3.2 AGGREGATER

FRML _I        IL            = FILA*PXA + FILN*PXN + FILE*PXE + FILI*PXI + FILT*PXT +
                               FILME*PME + FILMI*PMI + FILSV*PSIV ;
FRML _I        FIL           = IL/PIL ;

FRML _I        FIBPS         = (FIB*PIB[-1]-FIBOS*PIBOS[-1]-FIBH*PIBH[-1])/PIBPS[-1] ;
FRML _I        FIBPB         = FIBB+FIBI+FIBT ;
FRML _I        FIB           = (FIBA+FIBPB+FIBN+FIBE+FIBH+FIBS+FIBO)*KFIB ;
FRML _I        FIBXH         = (FIB*PIB[-1]-FIBH*PIBH[-1])/PIBXH[-1] ;
FRML _I        PIBXH         = (FIB*PIB-FIBH*PIBH)/FIBXH ;
FRML _I        IBXH          = FIBXH*PIBXH ;
FRML _I        FKBPB         = (FKBI + FKBB + FKBT) ;
FRML _I        FKB           = FKBPB + FKBA + FKBN + FKBE + FKBS + FKBH + FKBO ;

FRML _I        FIMPS         = (FIMX*PIMX[-1]-FIMXOS*PIMXOS[-1])/PIMPS[-1] ;
FRML _I        FIMPB         = FIMB+FIMI+FIMT ;
FRML _I        FIMX          = (FIMA+FIMPB+FIMN+FIME+FIMS+FIMXO)*KFIMX ;
FRML _I        FIM           = (FIMX*PIMX[-1]+FIOFU*PIOFU[-1])/PIM[-1] ;
FRML _I        FKMPB         = (FKMI + FKMB + FKMT) ;
FRML _I        FKPB          = FKMPB + FKBPB ;
FRML _I        FKM           = FKMPB + FKMA + FKMN + FKME + FKMS+ FKMO ;

FRML _I        FIF           = (FIM*PIM[-1]+FIB*PIB[-1])/PIF[-1] ;
FRML _I        FIFPS         = (FIF*PIF[-1]-FIOS*PIOS[-1])/PIFPS[-1] ;
FRML _I        PIFPS         = (FIF*PIF-FIOS*PIOS)/FIFPS ;
FRML _I        IFPS          = PIFPS*FIFPS ;
FRML _I        FI            = (FIF*PIF[-1]+FIL*PIL[-1])/PI[-1] ;

FRML _I        FIV           = DPRMA*FKMA[-1] + DPRMI*FKMI[-1] + DPRMT*FKMT[-1] + DPRMB*FKMB[-1]
                               +DPRME*FKME[-1] + DPRMS*FKMS[-1] + DPRMN*FKMN[-1] + DPRMO*FKMO[-1]
                               +DPRBA*FKBA[-1] + DPRBI*FKBI[-1] + DPRBT*FKBT[-1] + DPRBB*FKBB[-1]
                               +DPRBE*FKBE[-1] + DPRBS*FKBS[-1] + DPRBN*FKBN[-1] + DPRBO*FKBO[-1]
                               +DPRBH*FKBH[-1];

                                                     
// ************************************************  
// * 4. BOLIGINVESTERINGER OG KONTANTPRIS         *  
// ************************************************  
FRML _DJ_D     tsuih         = tsk + tss + tks ;
FRML _GJRD     bej           =  (1-tsuih*dse)*(Siqejh/phk)/(kfkbhe[-2]*fkbh[-2])
                               + (tsuih*tsdl+tqkej*kssyej)*(0.5*phv+ 0.5*phv[-1])/phk ;

FRML _D        ucost         = iwbz*(1-tsuih) + bej;
FRML _D        gpcpe         = (1-Dgpcpe)*( 0.5*gpcpe[-1] + 0.5*(pcp/pcp[-1]-1) + Jgpcpe )
                               + Dgpcpe*Zgpcpe ;

// Kontantprisrelation
FRML _DJRD     Log(bfkbhw)   = -0.48078*Log(phk/pcp)
                               -4.01561*Log(1+ucost+dprbh-gpcpe)
                               + kbfkbhw ;
FRML _SJRD     Dlog(phk)     = 3.23685*Dlog(q)
                               - 3.06451*Dlog(1+ucost+dprbh-gpcpe)
                               + 0.36172*Dlog(fkbh[-1])
                               + gphk
                               - 0.36172*(Log(fkbh[-1]/fydl[-1])-Log(bfkbhw[-1])) ;

// Realkreditgæld
FRML _GJR      wbh           = phk*fkbh*kfkbhe;

// Private boliginvesteringer
FRML _D        tobinqh       = phk/(pibh**0.7*phgk**0.3) ;

FRML _SJRD     finbh         = ( 0.04336*tobinqh + 0.01448*d4879 - 0.03032 )*fkbh[-1] + 1.0446*nbs ;

FRML _I        fibh          = finbh + dprbh*fkbh[-1];
FRML _I        fkbh          = fkbh[-1] + fibh/kfkbh - dprbh*fkbh[-1] ;

// Grundpris og vurdering
FRML _GJ       phgk          = phgk[-1]*phk/phk[-1] ;
FRML _GJ       phv           = 0.5*phk + 0.5*phk[-1] ;

                                                     
// ************************************************  
// * 5. IMPORT, MÆNGDER                           *  
// ************************************************  
// 5.1 Import af varer ekskl. energi
FRML _D        fami          = fami[-1]*(amia[-1]*fxa+amib[-1]*fxb+amii[-1]*fxi+amit[-1]*fxt
                                         +amicv[-1]*fcv+amiim[-1]*fimx+amiei[-1]*fei)/fmzi[-1] ;
FRML _DJRD     pxmi          = (pmi+tmi)/pxi ;

FRML _DJRD     Log(bfmziw)   = -1.90233*Log(pxmi) + kbfmziw ;
FRML _SJRD     Dlog(fmzi)    = 1.60218*Dlog(fami) - 0.7033*Dlog(pxmi) + gfmzi
                               -0.14996*(Log(fmzi[-1]/fami[-1])-Log(bfmziw[-1])) ;
FRML _D        fmi           = fmzi + amio*fxo + amicb*fcb + filmi ;
FRML _G        kfmi          = (fmzi/fami)/(fmzi[-1]/fami[-1]) ;

// 5.2 Import af tjenester ekskl. søfart og turisme
FRML _D        famsq         = famsq[-1]*( amsqb[-1]*fxb+amsqi[-1]*fxi+amsqt[-1]*fxt
                                          +amsqcs[-1]*fcs+amsqim[-1]*fimx+amsqesq[-1]*fesq)/fmzsq[-1] ;
FRML _DJRD     pxmsq         = pmsq/pxt ;

FRML _DJRD     Log(bfmzsqw)  = -0.48934*Log(pxmsq) - 0.29826*d4803 + 0.02365*dtfmzsq + kbfmzsqw ;
FRML _SJRD     Dlog(fmzsq)   = 0.66452*Dlog(famsq) - 0.39677*Dlog(pxmsq) + 0.01208*Diff(dtfmzsq) + gfmzsq
                               -0.51089*(Log(fmzsq[-1]/famsq[-1])-Log(bfmzsqw[-1])) ;
FRML _D        fmsq          = fmzsq + amsqo*fxo;
FRML _G        kfmsq         = (fmzsq/famsq)/(fmzsq[-1]/famsq[-1]);

// 5.3 Import af energi, turisme, søfart
FRML _D        fme           = amee*fxe + amet*fxt + ames*fxs + amecg*fcg + ameee*fee + filme;
FRML _I        fmt           = fct;
FRML _D        fmss          = amsss*fxs;

// 5.4 Aggregater
FRML _I        fmv           = (fme*pme[-1]+fmi*pmi[-1])/pmv[-1];
FRML _I        fms           = (fmss*pmss[-1]+fmsq*pmsq[-1])/pms[-1];
FRML _I        fmst          = (fms*pms[-1]+fmt*pmt[-1])/pmst[-1];
FRML _I        fm            = (fmv*pmv[-1]+fmst*pmst[-1])/pm[-1];

                                                     
// ************************************************  
// * 6.  IO-KOEFFICIENTER                         *  
// ************************************************  
// 6.1 IMPORTSUBSTITUTION, INDUSTRIVARER
FRML _G        AMIA          = AMIA[-1]*KFMI + J0AMIA ;
FRML _G        AMIB          = AMIB[-1]*KFMI + J0AMIB ;
FRML _G        AMII          = AMII[-1]*KFMI + J0AMII ;
FRML _G        AMIT          = AMIT[-1]*KFMI + J0AMIT ;
FRML _G        AMICV         = AMICV[-1]*KFMI + J0AMICV ;
FRML _G        AMIIM         = AMIIM[-1]*KFMI + J0AMIIM ;
FRML _G        AMIEI         = AMIEI[-1]*KFMI + J0AMIEI ;

// MODKORREKTION AF KOEFFICIENTER FOR INDENLANDSKE LEVERANCER FRA INDUSTRIEN
FRML _G        AIA           = (AIA[-1]+JAIA) - (AMIA[-1]*(KFMI-1))*(PMI[-1]/PXI[-1]) ;
FRML _G        AIB           = (AIB[-1]+JAIB) - (AMIB[-1]*(KFMI-1))*(PMI[-1]/PXI[-1]) ;
FRML _G        AII           = (AII[-1]+JAII) - (AMII[-1]*(KFMI-1))*(PMI[-1]/PXI[-1]) ;
FRML _G        AIT           = (AIT[-1]+JAIT) - (AMIT[-1]*(KFMI-1))*(PMI[-1]/PXI[-1]) ;
FRML _G        AICV          = (AICV[-1]+JAICV) - (AMICV[-1]*(KFMI-1))*(PMI[-1]/PXI[-1]) ;
FRML _G        AIIM          = (AIIM[-1]+JAIIM) - (AMIIM[-1]*(KFMI-1))*(PMI[-1]/PXI[-1]) ;
FRML _G        AIEI          = (AIEI[-1]+JAIEI) - (AMIEI[-1]*(KFMI-1))*(PMI[-1]/PXI[-1]) ;


// 6.2 IMPORTSUBSTITUTION, TJENESTER
FRML _G        AMSQB         = AMSQB[-1]*KFMSQ + J0AMSQB ;
FRML _G        AMSQI         = AMSQI[-1]*KFMSQ + J0AMSQI ;
FRML _G        AMSQT         = AMSQT[-1]*KFMSQ + J0AMSQT ;
FRML _G        AMSQCS        = AMSQCS[-1]*KFMSQ + J0AMSQCS ;
FRML _G        AMSQIM        = AMSQIM[-1]*KFMSQ + J0AMSQIM ;
FRML _G        AMSQESQ       = AMSQESQ[-1]*KFMSQ + J0AMSQESQ ;

// MODKORREKTION AF KOEFFICIENTER FOR INDENLANDSKE LEVERANCER FRA PRIVATE TJENESTEYDENDE ERHVERV
FRML _G        ATB           = (ATB[-1]+JATB) - (AMSQB[-1]*(KFMSQ-1))*(PMSQ[-1]/PXT[-1]) ;
FRML _G        ATI           = (ATI[-1]+JATI) - (AMSQI[-1]*(KFMSQ-1))*(PMSQ[-1]/PXT[-1]) ;
FRML _G        ATT           = (ATT[-1]+JATT) - (AMSQT[-1]*(KFMSQ-1))*(PMSQ[-1]/PXT[-1]) ;
FRML _G        ATCS          = (ATCS[-1]-((AOCS-AOCS[-1])*PXO[-1]+J0AMSQCS*PMSQ[-1]+JASVCS*PSIV[-1])/PXT[-1]-J0ATCS)
                                - (AMSQCS[-1]*(KFMSQ-1))*(PMSQ[-1]/PXT[-1]) ;
FRML _G        ATIM          = (ATIM[-1]-(JAIIM*PXI[-1]+J0AMIIM*PMI[-1]+J0AMSQIM*PMSQ[-1]+JASVIM*PSIV[-1])/PXT[-1]-J0ATIM)
                                - (AMSQIM[-1]*(KFMSQ-1))*(PMSQ[-1]/PXT[-1]) ;
FRML _G        ATESQ         = (ATESQ[-1]-(JABESQ*PXB[-1]+JAIESQ*PXI[-1]+J0AMSQESQ*PMSQ[-1]+JASVESQ*PSIV[-1])/PXT[-1]-J0ATESQ)
                               - (AMSQESQ[-1]*(KFMSQ-1))*(PMSQ[-1]/PXT[-1]) ;

// 6.3 IO-JUSTERINGER

// ENDELIGE ANVENDELSER
// For at sikre konsistens modposteres i én celle pr. søjle - som udgangspunkt leverancen fra T (omfatter bl.a. handelsavance)
FRML _G        AECE          = AECE[-1] + JAECE ;
FRML _G        ATCE          = ATCE[-1] -(JAECE*PXE[-1]+JASVCE*PSIV[-1])/PXT[-1]-J0ATCE ;
FRML _G        ASVCE         = ASVCE[-1] + JASVCE ;

FRML _G        AECG          = AECG[-1] -(JATCG*PXT[-1]+JAMECG*PME[-1]+JASVCG*PSIV[-1])/PXE[-1]-J0AECG ;
FRML _G        ATCG          = ATCG[-1] + JATCG ;
FRML _G        AMECG         = AMECG[-1] + JAMECG ;
FRML _G        ASVCG         = ASVCG[-1] + JASVCG ;

FRML _G        AHCH          = AHCH[-1] + JAHCH ;
FRML _G        ATCH          = ATCH[-1] -(JAHCH*PXH[-1]+JASVCH*PSIV[-1])/PXT[-1]-J0ATCH ;
FRML _G        ASVCH         = ASVCH[-1] + JASVCH ;

FRML _G        AACV          = AACV[-1] + JAACV ;
FRML _G        ATCV          = ATCV[-1] -(JAACV*PXA[-1]+JAICV*PXI[-1]+J0AMICV*PMI[-1]+JASVCV*PSIV[-1])/PXT[-1]-J0ATCV ;
FRML _G        ASVCV         = ASVCV[-1] + JASVCV ;

FRML _G        ASVCS         = ASVCS[-1] + JASVCS ;

FRML _G        ATCB          = ATCB[-1] -(JAMICB*PMI[-1]+JASVCB*PSIV[-1])/PXT[-1]-J0ATCB ;
FRML _G        AMICB         = AMICB[-1] + JAMICB ;
FRML _G        ASVCB         = ASVCB[-1] + JASVCB ;

FRML _G        ATCO          = ATCO[-1] -(JAOCO*PXO[-1]+JASVCO*PSIV[-1])/PXT[-1]-J0ATCO ;
FRML _G        AOCO          = AOCO[-1] + JAOCO ;
FRML _G        ASVCO         = ASVCO[-1] + JASVCO ;

FRML _G        ASVIM         = ASVIM[-1] + JASVIM ;

FRML _G        ABIB          = ABIB[-1] + JABIB ;
FRML _G        ATIB          = ATIB[-1] -(JABIB*PXB[-1]+JASVIB*PSIV[-1])/PXT[-1]-J0ATIB ;
FRML _G        ASVIB         = ASVIB[-1] + JASVIB ;

FRML _G        ANEE          = ANEE[-1] + JANEE ;
FRML _G        AEEE          = AEEE[-1] -(JANEE*PXN[-1]+JAMEEE*PME[-1]+JASVEE*PSIV[-1])/PXE[-1]-J0AEEE ;
FRML _G        AMEEE         = AMEEE[-1] + JAMEEE ;
FRML _G        ASVEE         = ASVEE[-1] + JASVEE ;

FRML _G        AAEI          = AAEI[-1] + JAAEI ;
FRML _G        ATEI          = ATEI[-1] -(JAAEI*PXA[-1]+JAIEI*PXI[-1]+J0AMIEI*PMI[-1]+JASVEI*PSIV[-1])/PXI[-1]-J0ATEI ;
FRML _G        ASVEI         = ASVEI[-1] + JASVEI ;

FRML _G        ABESQ         = ABESQ[-1] + JABESQ ;
FRML _G        AIESQ         = AIESQ[-1] + JAIESQ ;
FRML _G        ASVESQ        = ASVESQ[-1] + JASVESQ ;

// ERHVERVENES PRODUKTION
// For at sikre konsistens modposteres i én celle pr. søjle - som udgangspunkt BVT-indholdet

FRML _G        AAA           = AAA[-1] + JAAA ;
FRML _G        AEA           = AEA[-1] + JAEA ;
FRML _G        ATA           = ATA[-1] + JATA ;
FRML _G        ASVA          = ASVA[-1] + JASVA ;
FRML _G        AYFA          = AYFA[-1] -(JAAA*PXA[-1]+JAEA*PXE[-1]+JAIA*PXI[-1]+JATA*PXT[-1]+J0AMIA*PMI[-1]+JASVA*PSIV[-1])/PYFA[-1]-J0AYFA ;

FRML _G        AEB           = AEB[-1] + JAEB ;
FRML _G        ASVB          = ASVB[-1] + JASVB ;
FRML _G        AYFB          = AYFB[-1] -(JAEB*PXE[-1]+JAIB*PXI[-1]+JATB*PXT[-1]+J0AMIB*PMI[-1]+J0AMSQB*PMSQ[-1]+JASVB*PSIV[-1])/PYFB[-1]-J0AYFB ;

FRML _G        ATN           = ATN[-1] + JATN ;
FRML _G        ASVN          = ASVN[-1] + JASVN ;
FRML _G        AYFN          = AYFN[-1] -(JATN*PXT[-1]+JASVN*PSIV[-1])/PYFN[-1]-J0AYFN ;

FRML _G        AEE           = AEE[-1] + JAEE ;
FRML _G        ATE           = ATE[-1] + JATE ;
FRML _G        AMEE          = AMEE[-1] -((ANE-ANE[-1])*PXN[-1]+JAEE*PXE[-1]+JATE*PXT[-1]+JAYFE*PYFE[-1]+JASVE*PSIV[-1])/PME[-1]-J0AMEE ;
FRML _G        ASVE          = ASVE[-1] + JASVE ;
FRML _G        AYFE          = AYFE[-1] + JAYFE ;

FRML _G        ABH           = ABH[-1] + JABH ;
FRML _G        ATH           = ATH[-1] + JATH ;
FRML _G        ASVH          = ASVH[-1] + JASVH ;
FRML _G        AYFH          = AYFH[-1] -(JABH*PXB[-1]+JATH*PXT[-1]+JASVH*PSIV[-1])/PYFH[-1]-J0AYFH ;

FRML _G        AAI           = AAI[-1] + JAAI ;
FRML _G        AEI           = AEI[-1] + JAEI ;
FRML _G        ASVI          = ASVI[-1] + JASVI ;
FRML _G        AYFI          = AYFI[-1]-(JAAI*PXA[-1]+JAEI*PXE[-1]+JAII*PXI[-1]+JATI*PXT[-1]+J0AMII*PMI[-1]+J0AMSQI*PMSQ[-1]+JASVI*PSIV[-1])/PYFI[-1]-J0AYFI ;

FRML _G        ABT           = ABT[-1] + JABT ;
FRML _G        AET           = AET[-1] + JAET ;
FRML _G        AMET          = AMET[-1] + JAMET ;
FRML _G        ASVT          = ASVT[-1] + JASVT ;
FRML _G        AYFT          = AYFT[-1] -(JABT*PXB[-1]+JAET*PXE[-1]+JAIT*PXI[-1]+JATT*PXT[-1]+(AST-AST[-1])*PXS[-1]+JAMET*PME[-1]+J0AMIT*PMI[-1]+J0AMSQT*PMSQ[-1]+JASVT*PSIV[-1])/PYFT[-1]-J0AYFT ;

FRML _G        ATS           = ATS[-1] + JATS ;
FRML _G        AMES          = AMES[-1] + JAMES ;
FRML _G        AMSSS         = AMSSS[-1] + JAMSSS ;
FRML _G        ASVS          = ASVS[-1] + JASVS ;
FRML _G        AYFS          = AYFS[-1] -(JATS*PXT[-1]+JAMES*PME[-1]+JAMSSS*PMSS[-1]+JASVS*PSIV[-1])/PYFS[-1]-J0AYFS ;

FRML _G        AEO           = AEO[-1] + JAEO ;
FRML _G        ATO           = ATO[-1] -(JAEO*PXE[-1]+JAOO*PXO[-1]+JAMIO*PMI[-1]+JAMSQO*PMSQ[-1]+JASVO*PSIV[-1]+JAYFO*PYFO[-1])/PXT[-1]-J0ATO ;
FRML _G        AOO           = AOO[-1] + JAOO ;
FRML _G        AMIO          = AMIO[-1] + JAMIO ;
FRML _G        AMSQO         = AMSQO[-1] + JAMSQO ;
FRML _G        ASVO          = ASVO[-1] + JASVO ;
FRML _G        AYFO          = AYFO[-1] + JAYFO ;

// 6.4 SÆRBEHANDLEDE IO-KOEFFICIENTER
// SØFART
// IO-ligninger "vendes": PMSS er eneste eksogene pris, og fast forhold mellem FESS og FXS
FRML _G        AST           = (KFXS-1)*FESS/FXT ;

// NORDSØ
// IO-ligninger "vendes": FXN eksogen, FEE følger FXN
FRML _G        ANE           = (FXN-FILN-ANEE*FEE)/FXE ;

// OFFENTLIG PRODUKTION
// IO-ligninger "vendes": AOCS*FCS følger FXO, så en ændring i FCS ikke ændrer FXO
FRML _G        AOCS          = KOCS*FXO/FCS ;

// IO-KOEFFICIENTER VEDR. OFFENTLIGT VAREKØB
//EQ  ATO              = ATO[-1]  + ATO[-1]*KFVO + JATO ;
//EQ  AOO              = AOO[-1]  + AOO[-1]*KFVO + JAOO ;
//EQ  ASVO             = ASVO[-1] + ASVO[-1]*KFVO + JASVO ;
//EQ  AMIO             = AMIO[-1] + AMIO[-1]*KFVO + JAMIO ;
//EQ  AEO              = AEO[-1]  + AEO[-1]*KFVO + JAEO ;
// Tænk lige over om ovenstående kan bruges i stedet for den foreslåede residualberegning af ATO... /10.02.15 DG

                                                     
// ************************************************  
// * 7.  PRODUKTIONSVÆRDI, BVT MV., MÆNGDER       *  
// ************************************************  
// 7.1 PRODUKTIONSVÆRDI I PRIVATE ERHVERV
FRML _G        FXA           = (AAI*FXI + AACV*FCV + AAEI*FEI + FILA)/(1-AAA);
FRML _G        FXB           = ABH*FXH + ABT*FXT + ABIB*FIB + ABESQ*FESQ;
FRML _G        FXE           = (AEA*FXA + AEB*FXB + AEI*FXI + AET*FXT + AEO*FXO + AECE*FCE + AECG*FCG + AEEE*FEE + FILE)/(1-AEE);
FRML _G        FXH           = AHCH*FCH;
FRML _G        FXI           = (AIA*FXA + AIB*FXB + AIT*FXT + AICV*FCV + AIIM*FIMX + AIEI*FEI + AIESQ*FESQ + FILI)/(1-AII);
FRML _G        FXT           = (ATA*FXA + ATB*FXB + ATN*FXN + ATE*FXE + ATH*FXH + ATI*FXI + ATS*FXS + ATO*FXO +
                                ATCE*FCE + ATCG*FCG + ATCH*FCH + ATCV*FCV + ATCS*FCS + ATCB*FCB + ATCO*FCO +
                                ATIM*FIMX + ATIB*FIB + ATEI*FEI + ATESQ*FESQ + FILT)
                                /(1-ATT);
FRML _G        FXS           = KFXS*FESS;

// 7.2 BVT I PRIVATE ERHVERV
FRML _G        FYFA          = FXA*AYFA;
FRML _G        FYFB          = FXB*AYFB;
FRML _G        FYFN          = FXN*AYFN;
FRML _G        FYFE          = FXE*AYFE;
FRML _G        FYFH          = FXH*AYFH;
FRML _G        FYFI          = FXI*AYFI;
FRML _G        FYFT          = FXT*AYFT*KKF;
FRML _G        FYFS          = FXS*AYFS;

// 7.3 VAREKØB I PRIVATE ERHVERV
// ENERGI
FRML _G        FVEA          = FXA*AEA;
FRML _G        FVEB          = FXB*AEB;
FRML _G        FVEE          = FXE*(ANE+AEE+AMEE);
FRML _G        FVEI          = FXI*AEI;
FRML _G        FVET          = FXT*(AET+AMET);
FRML _G        FVES          = FXS*AMES;

// MATERIALER  inkl. alle varetilknyttede afgifter i erhvervet realt
FRML _G        FVMA          = FXA*(AAA+AIA+ATA+AMIA+ASVA);
FRML _G        FVMB          = FXB*(AIB+ATB+AMIB+AMSQB+ASVB);
FRML _G        FVMN          = FXN*(ATN+ASVN);
FRML _G        FVME          = FXE*(ATE+ASVE);
FRML _G        FVMH          = FXH*(ABH+ATH+ASVH);
FRML _G        FVMI          = FXI*(AAI+AII+ATI+AMII+AMSQI+ASVI);
FRML _G        FVMT          = FXT*(ABT+AIT+ATT+AST+AMIT+AMSQT+ASVT);
FRML _G        FVMS          = FXS*(ATS+AMSSS+ASVS);


// 7.4 AGGREGATER
FRML _I        FXPB          = (FXB*PXB[-1]+FXI*PXI[-1]+FXT*PXT[-1])/PXPB[-1];
FRML _I        PXPB          = XPB/FXPB;
FRML _I        FXPR          = (FXA*PXA[-1]+FXE*PXE[-1]+FXN*PXN[-1]+FXH*PXH[-1]+FXS*PXS[-1])/PXPR[-1];
FRML _I        PXPR          = XPR/FXPR;
FRML _I        FXP           = (FXPB*PXPB[-1]+FXPR*PXPR[-1])/PXP[-1];
FRML _I        PXP           = XP/FXP;
FRML _I        FX            = (FXP*PXP[-1]+FXO*PXO[-1])/PX[-1];

FRML _I        FYFPB         = (FYFB*PYFB[-1]+FYFI*PYFI[-1]+FYFT*PYFT[-1])/PYFPB[-1];
FRML _I        FYFPR         = (FYFA*PYFA[-1]+FYFE*PYFE[-1]+FYFN*PYFN[-1]+FYFH*PYFH[-1]+FYFS*PYFS[-1])/PYFPR[-1];
FRML _I        PYFPR         = YFPR/FYFPR;
FRML _I        FYFR          = (FYFPR*PYFPR[-1]+FYFO*PYFO[-1])/PYFR[-1];
FRML _I        FYFP          = (FYFPB*PYFPB[-1]+FYFPR*PYFPR[-1])/PYFP[-1];
FRML _I        FYF           = (FYFP*PYFP[-1]+FYFO*PYFO[-1])/PYF[-1];
FRML _I        FVEX          = FVEA + FVEB + FVEI + FVEE + FVET + FVES + FVEO;
FRML _I        FVX           = (FX*PX[-1]-FYF*PYF[-1])/PVX[-1];

FRML _I        FSIV          = ASVA*FXA + ASVB*FXB + ASVN*FXN + ASVE*FXE + ASVH*FXH + ASVI*FXI + ASVT*FXT + ASVS*FXS + ASVO*FXO +
                               ASVCE*FCE + ASVCG*FCG + ASVCH*FCH + ASVCV*FCV + ASVCS*FCS + ASVCB*FCB + ASVCO*FCO +
                               ASVIM*FIMX + ASVIB*FIB + ASVEE*FEE + ASVEI*FEI + ASVESQ*FESQ + FILSV;

FRML _I        FAT           = (FYF*PYF[-1]+FM*PM[-1]+FSIV*PSIV[-1])/PAT[-1];
FRML _I        FAI           = (FCP*PCP[-1]+FCO*PCO[-1]+FI*PI[-1])/PAI[-1];
FRML _I        FAIPS         = (FAI*PAI[-1]-FCO*PCO[-1]-FIOS*PIOS[-1])/PAIPS[-1];
FRML _I        FAE           = (FAI*PAI[-1]+FE*PE[-1])/PAE[-1];

FRML _I        FY            = (FAE*PAE[-1]-FM*PM[-1])/PY[-1];

// MÆNGDEKORREKTION
FRML _G        KKF           = KKF + 1 - FAT/FAE;

                                                     
// ************************************************  
// * 8. BESKÆFTIGELSE, ARBEJDSTID, ARBEJDSSTYRKE, *  
// *    LEDIGHED OG PRODUKTIVITET                 *  
// ************************************************  
// 8.1 BESKÆFTIGELSE I TIMER

// I-ERHVERV
FRML _GJRD    Log(hqiw)     = (dtalfai-1)*Log((1-dtalfai)/dtalfai)
                               + Log(fyfi) - Log(dtai)
                               + (dtalfai-1)*(Log(li)-Log(pkzi));
FRML _GJRD    hqin          = EXP(-(1/dtalfai)*LOG(dtai)
                                   +(1/dtalfai)*LOG(fyfi)
                                   -((1-dtalfai)/dtalfai)*LOG(fkzi));
FRML _S__D    Log(hqi)      = (1-DXVYFHI)*(0.50458*Log(hqin) + 0.49542*Log(hqin[-1]) + jrhqi)
                                + DXVYFHI*Log(fyfi/vyfhix);

// T-ERHVERV
FRML _GJRD    Log(hqtw)     = (dtalfat-1)*Log((1-dtalfat)/dtalfat)
                               + Log(fyft) - Log(dtat)
                               + (dtalfat-1)*(Log(lt)-Log(pkzt));
FRML _GJRD    hqtn          = EXP(-(1/dtalfat)*LOG(dtat)
                                   +(1/dtalfat)*LOG(fyft)
                                   -((1-dtalfat)/dtalfat)*LOG(fkzt));
FRML _S__D    Log(hqt)      = (1-DXVYFHT)*(0.61527*Log(hqtn) + 0.38473*Log(hqtn[-1]) + jrhqt)
                               + DXVYFHT*Log(fyft/vyfhtx);

// B-ERHVERV
FRML _GJRD     Log(hqbw)     = (dtalfab-1)*Log((1-dtalfab)/dtalfab)
                               + Log(fyfb) - Log(dtab)
                               + (dtalfab-1)*(Log(lb)-Log(pkzb));
FRML _GJRD     hqbn          = EXP(-(1/dtalfab)*LOG(dtab)
                                   +(1/dtalfab)*LOG(fyfb)
                                   -((1-dtalfab)/dtalfab)*LOG(fkzb));
FRML _S__D     Log(hqb)      = (1-DXVYFHB)*(0.80208*Log(hqbn) + 0.19792*Log(hqbn[-1]) + jrhqb)
                                + DXVYFHB*Log(fyfb/vyfhbx);

// ØVRIGE ERHVERV OG AGGREGATER
FRML _G        HQA           = (1-DXVYFHA)*HQAX + DXVYFHA*FYFA/VYFHAX ;
FRML _D        HQS           = QWS*HGS/1000 ;
FRML _D        HQO           = QO*HGO/1000 ;
FRML _D        HQPB          = HQT + HQI + HQB ;
FRML _D        HQPR          = HQA + HQN + HQE + HQH + HQS ;
FRML _D        HQR           = HQPR + HQO ;
FRML _D        HQ            = HQPB + HQR ;

// 8.2 BESKÆFTIGELSE I PERSONER
FRML _D        QA            = HQA/HGA*1000 ;
FRML _D        QSA           = BQSA*QA ;
FRML _D        QWA           = QA-QSA ;
FRML _D        QB            = HQB/HGB*1000 ;
FRML _D        QSB           = BQSB*QB ;
FRML _D        QWB           = QB-QSB ;

FRML _D        QE            = HQE/HGE*1000 ;
FRML _D        QSE           = BQSE*QE ;
FRML _D        QWE           = QE-QSE ;
FRML _D        QN            = HQN/HGN*1000 ;
FRML _D        QSN           = BQSN*QN ;
FRML _D        QWN           = QN-QSN ;
FRML _D        QH            = HQH/HGH*1000 ;
FRML _D        QSH           = BQSH*QH ;
FRML _D        QWH           = QH-QSH ;

FRML _D        QI            = HQI/HGI*1000 ;
FRML _D        QSI           = BQSI*QI ;
FRML _D        QWI           = QI-QSI ;

FRML _D        QT            = HQT/HGT*1000 ;
FRML _D        QST           = BQST*QT ;
FRML _D        QWT           = QT-QST ;

FRML _D        QWO           = QO - QSO ;
FRML _D        QWP           = QWN + QWI + QWA + QWH + QWB + QWE + QWT + QWS ;
FRML _D        QW            = QWP + QWO ;

FRML _D        QPS           = Q - QOS ;
FRML _D        QPB           = QI + QB + QT ;

FRML _D        QS            = QSA + QSH + QSB + QSE + QSN + QSI + QST + QSO ;
FRML _D        Q             = QW + QS ;

FRML _GJ_D     Dif(Qm)       = Dif(QMB + QMF + QMS + QMR) ;
FRML _I        Q1            = Q + QM ;

// 8.3 GENNEMSNITLIG ARBEJDSTID
FRML _G        HGW           = KHGW*HA ;
FRML _G        HGA           = KHGA*HGW ;
FRML _G        HGB           = KHGB*HGW ;
FRML _G        HGE           = KHGE*HGW ;
FRML _G        HGH           = KHGH*HGW ;
FRML _G        HGN           = KHGN*HGW ;
FRML _G        HGI           = KHGI*HGW ;
FRML _G        HGO           = KHGO*HGW ;
FRML _G        HGT           = KHGT*HGW ;
FRML _G        HGS           = KHGS*HGW ;

FRML _D        HG            = HQ/Q*1000 ;

// 8.4 TIMEPRODUKTIVITET
FRML _D        VYFHB         = FYFB/HQB ;
FRML _D        VYFHI         = FYFI/HQI ;
FRML _D        VYFHT         = FYFT/HQT ;
FRML _D        VYFHPB        = FYFPB/HQPB ;
FRML _D        VYFHA         = FYFA/HQA ;
FRML _D        VYFHE         = FYFE/HQE ;
FRML _D        VYFHN         = FYFN/HQN ;
FRML _D        VYFHH         = FYFH/HQH ;
FRML _D        VYFHO         = FYFO/HQO ;
FRML _D        VYFHR         = FYFR/HQR ;
FRML _D        VYFH          = FYF/HQ ;

//     8.5 ARBEJDSTYRKE, LEDIGHED, AKTIVEREDE
FRML _D        UA            = UA_S + UA_K + JUA ;

// *** Personer på indkomstoverførsler ***
// Ledige
FRML _D        UL            = UA - Q ;
FRML _G        ULU           = KULU*UL ;
FRML _D        ULDP          = UL - ULU ;
FRML _D        ULDPD         = ULDP - ULDPA ;
FRML _D        ULBAK         = (UAKX+QLTJD)+BULBAK*(UKAK+QLTJK) ;
FRML _D        ULB           = UL + ULBAK ;

// Ledighedsrate
FRML _D        BUL           = UL/UA ;
FRML _D        BUL_S         = UL_S/UA_S ;

// Aktiverede uden for arbejdsstyrken
FRML _D        UAKX          = UAKX_S + UAKX_K ;

// Faktisk antal i konjunkturfølsomme grupper
FRML _D        UUXA          = UUXA_S + UUXA_K ;
FRML _D        UKXA          = UKXA_S + UKXA_K ;
FRML _D        UKAK          = UKAK_S + UKAK_K ;
FRML _D        ULY           = ULY_S + ULY_K ;
FRML _D        UPFO          = UPFO_S + UPFO_K ;
FRML _D        UFO           = UPFO_S + QPFO_S + UPFO_K + QPFO_K ;
FRML _D        UFP           = UPFP + QPFP ;
FRML _D        USS           = UMS + QMS ;
FRML _D        USB           = UMB + QMB ;
FRML _D        UMJ           = UMJ_S + UMJ_K ;

// Restgruppe ("Skovfolk")
FRML _D        UR_S          = U-(UA_S+UB+USS+USB+USF+UMR+UFDP+UKXA_S+UPFO_S+UUXA_S+UEF+UOV+UFY+ULY_S+URY+UAKX_S+UKAK_S+UPFP+UPT+UMJ_S) ;
FRML _D        UR_K          = -(UA_K+UUXA_K+UAKX_K+UKAK_K+UKXA_K+UPFO_K+ULY_K+UMJ_K) ;
FRML _D        UR            = UR_S + UR_K - JUA ;

// i beskæftigelse
FRML _G        QLTJD         = BQLTJD*(ULDP+ULDP[-1])/2 ;
FRML _G        QLTJK         = BQLTJK*(ULU+ULU[-1])/2 ;
FRML _G        QPFO_K        = -0.5*UPFO_K ;

                                                     
// ************************************************  
// * 9. DEN OFFENTLIGE SEKTOR                     *  
// ************************************************  
// 9.1 OFFENTLIG PRODUKTION, FORBRUG OG INVESTERINGER
// PRODUKTION, BVT OG FORBRUG, MÆNGDER
FRML _G        FXO           = (AOCO*FCO + FIOFU)/(1-AOO-KOCS) ;

FRML _G        FYFO          = FXO*AYFO ;
FRML _G        QO            = (FYFO-FIVO)/(KFYFO*HGO) ;
FRML _G        QOS           = KQOS*QO ;

FRML _G        FVEO          = FXO*AEO ;
FRML _G        FVMO          = FXO*(ATO+AOO+AMIO+AMSQO+ASVO) ;
FRML _G        FCOC          = KCOC*FCO ;
FRML _G        FCOI          = (1-KCOC)*FCO ;

FRML _GJR      FCOGL         = FCOGL[-1]*(FCO/FCO[-1]) ;
FRML _D        PCOGL         = CO/FCOGL ;

// Offentligt forbrug ekskl. afskrivninger (FCOZ) - tilføjet 01.05.19 DG
FRML _D        Coz           = Co - Ivos ;
FRML _D        fCo           = (fCoz*pcoz(-1)+fIvos*pivos(-1))/pco(-1) ;
FRML _D        pcoz          = Coz/fCoz ;
//ML _D        FIVOS         = KFIVOS*FIVO ;
FRML _D        FIVOS         = DXIVOS*(IVOS/PIVOS) + (1-DXIVOS)*KFIVOS*FIVO ;  // Tilføjet swift-mulighed mellem rent modelbestemt (DXIVOS=0) eller udfra bud på nom. afskr. (DXIVOS=1) - var gjort direkte i SMEC.FRM til F19 /24.06.19 DG


// PRODUKTION, BVT OG FORBRUG, VÆRDIER
FRML _G        YFO           = YWO + IVOS*KIVO + SIQO ;
FRML _G        VOS           = VO*KVOS ;
FRML _G        YWOS          = YWO*KYWOS ;
FRML _G        YFOS          = YWOS + IVOS + SIQO*KSIQOS ;
FRML _D        XOS           = YFOS + VOS ;

FRML _D        XO            = YFO + VO ;
FRML _D        CO            = PCO*FCO ;
FRML _G        COC           = KCOC*CO ;
FRML _G        COI           = (1-KCOC)*CO ;
FRML _G        VO            = FXO*(AEO*PXE + ATO*PXT + AOO*PXO + AMIO*(PMI+TMI) + AMSQO*PMSQ)*KPVO + SIPVO + SIGVO ;

// INVESTERINGER, VÆRDI OG MÆNGDE
// Offentligt produceret forskning og udvikling
FRML _G        IOFU          = (1-DXIOFU)*(TIOFU*Y_S + JIOFU) + DXIOFU*IOFUX ;
FRML _G        PIOFU         = PXO*KPIOFU ;
FRML _D        FIOFU         = IOFU/PIOFU ;

FRML _D        FIVO          = (DPRMO*FKMO[-1] + DPRBO*FKBO[-1]) ;
//ML _G        IVOS          = DXIVOS*BIVOS*CO + (1-DXIVOS)*PIVOS*FIVO ;
FRML _G        IVOS          = DXIVOS*BIVOS*CO + (1-DXIVOS)*PIVOS*FIVOS ;

FRML _D        FIOS          = (FIXOS*PIXOS[-1]+FIOFU*PIOFU[-1])/PIOS[-1] ;
FRML _G        FIBOS         = BIBOS*FIXOS ;

FRML _D        FIMXOS        = (FIXOS*PIXOS[-1]-FIBOS*PIBOS[-1])/PIMXOS[-1] ;
FRML _D        FIMOS         = (FIMXOS*PIMXOS[-1]+FIOFU*PIOFU[-1])/PIMOS[-1] ;


// 9.2 NETTOFORDRINGSERHVERVELSE OG RENTESTRØMME
FRML _D        TFOPN         = TFOIP - TFOUP ;
FRML _D        TFON          = TFOPN + TIION ;
FRML _D        TFOIP         = SD + (SI-SIM-SISU) + SA + TFOIQ ;
FRML _D        TFOIQ         = IVOS + TYPRI + TBPHO +  TIOV + TIOR + TEUR
                              + TAPO + TAFO + TKPO + TKFO ;

FRML _D        TFOUP         = CO + TY + IOS - SISUDK + TFOUQ ;
FRML _D        TFOUQ         = TEUBZ + TAOP + TAOF + TKOP + TKOF ;

FRML _D        TION          = TI_Z_O - TI_O_Z ;
FRML _D        TIION         = TIR_Z_O - TI_O_Z ;
FRML _D        TI_Z_O        = TIR_Z_O + TIOV + TIOR ;
FRML _GJ_D     Tir_z_o       = (iwb10ys+rpiw_z_o)*W_z_o[-1] ;                                        //Nye ligninger på foranledning af JMP+DG /02.05.18 RRS+DG
FRML _GJ_D     Ti_o_z        = (iwb10ys+rpiw_o_z)*W_o_z[-1] ;                                        //Nye ligninger på foranledning af JMP+DG /02.05.18 RRS+DG
FRML _D        TIOV          = TIOVR + TIOVN ;
FRML _D        TIOR          = TIORR + TIORN + TIRO + TIRK ;

FRML _G        W_z_o         = (Y/Y[-1]-1)*W_z_o[-1] + kW_z_o + Ow_z_o+w_z_o[-1] ;
FRML _G        W_o_z         = (Diff(W_z_o)-Ow_z_o)-tfon + Ow_o_z+w_o_z[-1] ;
FRML _D        WN_O          = W_z_o - W_o_z ;

FRML _GJ_      EMUGLD        = EMUGLD[-1] + DIFF(W_o_z) - Ow_o_z ;

// 9.3 OFFENTLIGE TRANSFERERINGER MV.
// Indkomstoverførlser i alt
FRML _D        TY            = TYP + TYD + TYM + TYU + TYR ;
FRML _D        Ty_off10      = Ty + Taopi + Taopy ;

// Tilbagetrukne
FRML _D        TYPFP         = PSRTY*TTYPFP*UFP*0.001 ;
FRML _D        TYPFO         = PSRTY*TTYPFO*UFO*0.001 ;
FRML _D        TYPE          = PSRTY*TTYPE*(UEF+UOV+UFY)*0.001 ;
FRML _D        TYPR          = PSRTY*TYPRX ;
FRML _D        TYP           = TYPFP + TYPFO + TYPE + TYPR + TYPFU ;

// Dagpenge
FRML _D        TYDD          = PSRTY*TTYDD*ULDPD*0.001 ;
FRML _D        TYDA          = PSRTY*TTYDA*ULDPA*0.001 ;
FRML _D        TYD           = TYDD + TYDA ;

// Midlertidigt udenfor arbejdsmarkedet
FRML _D        TYMS          = PSRTY*TTYMS*USS*0.001 ;
FRML _D        TYMB          = PSRTY*TTYMB*USB*0.001 ;
FRML _D        TYMF          = PSRTY*TTYMF*USF*0.001 ;
FRML _D        TYMR          = PSRTY*TTYMR*UMR*0.001 ;
FRML _D        TYMFDP        = PSRTY*TTYMFDP*UFDP*0.001 ;
FRML _D        TYM           = TYMS + TYMB + TYMF + TYMR + TYMFDP ;

// Uddanelse, aktivering mv.
FRML _D        TYUAK         = PSRTY*TTYUAK*UAKX*0.001 ;
FRML _D        TYUKAK        = PSRTY*TTYUKAK*UKAK*0.001 ;
FRML _D        TYULY         = PSRTY*TTYULY*ULY*0.001 ;
FRML _D        TYURY         = PSRTY*TTYURY*URY*0.001 ;
FRML _D        TYUSU         = PSRTY*TTYUSU*USU*0.001 ;
FRML _D        TYU           = TYULY + TYUAK + TYURY + TYUSU + TYUKAK ;

// Øvrige skattepligtige indkomstoverførsler (kontanthjælp mv.)
FRML _D        TYKSL         = PSRTY*TTYKS*ULU*0.001 ;
FRML _D        TYKSR         = PSRTY*TTYKS*UKXA*0.001 ;
FRML _D        TYKS          = TYKSL + TYKSR ;
FRML _D        TYRS          = PSRTY*TYRSX ;

// Ikke-skattepligtige indkomstoverførsler
FRML _D        TYKR          = PSRTY*TYKRX ;
FRML _D        TYRBF         = PTTYP*TTYRBF*UB*0.001 ;
FRML _D        TYRGC         = TTYRGC*(U-Ub)*0.001 ;
FRML _D        TYRHS         = PSRTY*TYRHSX ;
FRML _D        TYRHY         = PSRTY*TTYRHY*(UFP+UFO)*0.001 ;
FRML _D        TYRRR         = PSRTY*TYRRRX ;
FRML _D        TYRR          = TYKR + TYRBF + TYRHS + TYRHY + TYRRR + TYRGC ;
FRML _D        TYR           = TYKS + TYRS + TYRR ;

// Satsreguleringsindeks og pristalsreguleringsindeks
FRML _G        PSRTY         = DXPSRTY*PSRTYX + (1-DXPSRTY)*(PSRTY[-1]*
                               (((LNAP[-2]*(1-SAP[-2])*HA[-2])/
                               (LNAP[-3]*(1-SAP[-3])*HA[-3]))*
                               (1-TSDA)/(1-TSDA[-1]) + JPSRTY)) ;
FRML _G        PTTYP         = DXPTTYP*PTTYPX + (1-DXPTTYP)*PTTYP[-1]*(PCP[-2]/PCP[-3] + JPTTYP) ;

// Øvrige offentlige udgifter
FRML _GJ_D     Taopi         = Taopix*psrty ;
FRML _GJ_D     Taopy         = Taopyx*psrty ;
FRML _GJ_D     TAOPM         = PSRTY*TTAOPM*BTAOPM*(UFP+UFO)*0.001;  // Mediecheck til pensionister /30.10.18 DG
FRML _GJ_D     Taopr         = ttaopr*Y_s ;
FRML _D        Taop          = Taopi + Taopy + Taopm + Taopr ;

FRML _GJ_D     TKOPEF        = PSRTY*TTKOPEF*BTKOPEF*(BTPEF*UA)*0.001 ;
FRML _GJ_D     TKOPR         = TTKOPR*Y_S ;
FRML _D        TKOP          = TKOPR + TKOPEF ;

// 9.4 SAMLEDE OG DIREKTE SKATTER
FRML _D        S             = SD + SI - SISU + SA ;
FRML _D        SD            = SDK + SDV + SDU + SDA + SDP + SDS + SDR + SDM ;
FRML _D        SDK           = SSY + SSFK ;
FRML _D        SDP           = TSDPK*KSDP*TPPKU + SDPA ;

FRML _GJ       Tippps        = -(iwbz+iwbb)*Wbh-bfisim*Yw1+(iwdeh+0.004)*Yw1 ;
FRML _G        Ylws          = Saso + Tpaf + Tpef + kylws*psrty*Q ;
FRML _G        Yrphs         = kyrphs*tsdl*fKbh[-2]*phv[-1] ;

FRML _G        Ysp           = KYSP*((YW1+TWEN-TYPRI) + (TY-TYRR-TYPFU) + (TPPLU-(TPPIL+TPPIK))) - SDA ;
FRML _G        Ys            = kys*(Ysp + Yrphs + Tippps) - Ylws ;

FRML _D        btippps       = Tippps/Ys ;
FRML _GJ_D     bysk          = bysk_s + 0.11*(btippps-btippps_s) ;
FRML _GJ_D     byss          = byss_s + 0.11*(btippps-btippps_s) ;
FRML _DJ       tssp0         = bysb*tsb + bysm*tsm + byst*tst ;
FRML _DJ       tss0          = bysk*tsk + byss*tss ;

FRML _GJ_D     Ssyn          = (1-d4711)*(0.08-tss)*btipppsn*Tippps ;

FRML _G        Ssysp         = tssp0*Ysp*kssysp - Ssyn ;
FRML _G        Ssys          = tss0*Ys*kssys ;

FRML _G        SSYEJ         = (1-DXSSYEJ)*(KSSYEJ*TQKEJ*PHV*(KFKBHE[-1]*FKBH[-1]/0.8)) + DXSSYEJ*SSYEJX ;

FRML _GJ_D     SSYD          = TSSYD*Y_S ;
FRML _GJ_D     SSYA          = TSSYA*Y_S ;
FRML _GJ_D     SSYV          = TSSYV*Y_S ;

FRML _D        SSY           = SSYS + SSYSP + SSYD + SSYA + SSYEJ + SSYV ;

FRML _G        SDV           = TSDV*FKCB[-1] ;
FRML _D        SDU           = TSDU*QW ;

FRML _G        YSDA          = KYSDA*(YW1-TYPRI) ;
FRML _D        SDA           = TSDA*YSDA ;

FRML _G        SDM           = KSDM*TSDM*PSRTY*(U-UB)*0.001 ;

FRML _G        IVSPS         = (BIVM*PIMPS*FIMPS + BIVB*PIBPS*FIBPS)*KIVSPS ;
FRML _GJ_D     TIPPS         = 0.1*TIIPQN + TTIPPS*Y_S ;
FRML _G        YSDSQ         = KYSDSQ*(YF-YW1-SIQ-IVOS-YRH) + TIPPS - 0.75*IVSPS ;
FRML _GJ_D     SDSQ          = TSDS*YSDSQ ;
FRML _GJ_D     SN            = TSN*FXN*(PWOIL/80.07)*(VUSA/562.33963) ;
FRML _G        SDSN          = (1-KTIRO-KTIORN-KTIOVN-KTIRK)*SN ;
FRML _G        SDS           = SDSQ + SDSN ;

FRML _G        TIRO          = KTIRO*SN ;
FRML _G        TIORN         = KTIORN*SN ;
FRML _G        TIOVN         = KTIOVN*SN ;

FRML _G        TIRK          = KTIRK*SN ;
FRML _GJ_D     SDR           = (TSDR*BSDR*TIPPN + BSDR*TSDRA*OPP) ;

// 9.5 INDIREKTE OG ANDRE SKATTER
FRML _D        SI            = SIM + SIP + SIG + SIR + SIQ ;
FRML _D        SIV           = SIM + SIP + SIG + SIR ;
FRML _D        SIM           = FME*TME + FMI*TMI ;

FRML _GJ_D     Sipaf_pso     = ksippso*( Sippso+Siqpso)*1.04 ;
FRML _D        tpce          = tpcex  + (0.25*Sippso - 0.33*Sipaf_pso )/fCe ;
FRML _D        tpvei         = tpveix + (0.25*Sippso - 0.33*Sipaf_pso )/fVei ;
FRML _D        tpvet         = tpvetx + (0.25*Sippso - 0.33*Sipaf_pso )/fVet ;

FRML _D        Sip           = Sipv + Sipef ;
FRML _D        SIPEF         = (1-DXSIP)*(TPCE*FCE+TPCG*FCG)
                               +DXSIP*(PNCP/PNCP[-1])*(TPCE*FCE+TPCG*FCG)
                               +TPCB*FCB + TPCV*FCV + TPCH*FCH + TPCS*FCS
                               +TPCO*FCO
                               +TPIBPS*FIBPS + TPIMPS*FIMPS
                               +TPIMOS*FIMXOS + TPIBOS*FIBOS
                               +TPIBH*FIBH + TPIL*FIL
                               + SIPEI + SIPUR ;

FRML _D        SIPEI         = SIPEE + SIPEQ + .25*Sippso ;

FRML _D        SIG           = SIGC + SIGIN + SIGV ;
FRML _D        SIGC          = BTGCE*TG*FCE*(PCE/(1+BTGCE*TG)) +
                               BTGCG*TG*FCG*(PCG/(1+BTGCG*TG)) +
                               BTGCB*TG*FCB*(PCB/((1+TRCB)*(1+BTGCB*TG))) +
                               BTGCV*TG*FCV*(PCV/(1+BTGCV*TG)) +
                               BTGCH*TG*FCH*(PCH/(1+BTGCH*TG)) +
                               BTGCS*TG*FCS*(PCS/(1+BTGCS*TG)) +
                               BTGCO*TG*FCO*(PCO/(1+BTGCO*TG)) ;
FRML _D        SIGIN         = BTGIBH*TG*FIBH*(PIBH/(1+BTGIBH*TG)) +
                               BTGIMPS*TG*FIMPS*(PIMPS/((1+TRIMPS)*(1+BTGIMPS*TG))) +
                               BTGIMOS*TG*FIMXOS*(PIMXOS/(1+BTGIMOS*TG)) +
                               BTGIBOS*TG*FIBOS*(PIBOS/(1+BTGIBOS*TG)) +
                               BTGIBPS*TG*FIBPS*(PIBPS/(1+BTGIBPS*TG)) +
                               BTGIL*TG*FIL*(PIL/(1+BTGIL*TG)) ;
FRML _D        Sir           = trcb*fCb*(pcb/(1+trcb)) + trco*fCo*(pco/(1+trco)) + trimps*fImps*(pimps/(1+trimps)) ;

FRML _D        SIQ           = SIQU + SIQAB + SIQEJ + SIQV + SIQAM + SIQCO2 + SIQR + SIQSU + JSIQ ;

FRML _G        SIQU          = TQU*QW*0.001 ;
FRML _G        SIQAB         = TQAB*KSIQAB*YW ;
FRML _G        SIQAM         = KSIQAM*(YWT+YWH);
FRML _G        SIQEJ         = (1-DXSIQEJ)*((TSIQEJ*KSIQEJ)*PHV[-1]) + DXSIQEJ*SIQEJX ;
FRML _GJ_D     SIQV          = TSIQV*Y_S ;
FRML _D        SIQSU         = SIQSQ + SIQLT + SIQAA + SIQPSO ;
FRML _GJ_D     SIQSH         = TSIQSH*Y_S ;
FRML _GJ_D     SIQSR         = TSIQSR*Y_S ;
FRML _GJ_D     SIQEUR        = TSIQEUR*Y_S ;
FRML _D        SIQSQ         = SIQSR + SIQSH + SIQEUR ;
FRML _D        SIQLT         = PSRTY*TSIQLT*(QLTJD+QLTJK+QLTFS+UMJ)*0.001 ;

FRML _GJ_D     SIPUR         = TSIPUR*Y_S ;
FRML _D        SIPSU         = SIPUR + (SIPAA + SIPSUER) + SIPEE + SIPPSO ;
FRML _D        SISU          = SIPSU + SIQSU ;
FRML _D        SISUDK        = (SIPUR + SIPPSO) + (SIQSH + SIQSR + SIQLT + SIQPSO) ;
FRML _D        SISUEU        = (SIPEE + SIPAA + SIPSUER) + SIQEU ;

FRML _D        SASO          = SAQWY + SAR ;

FRML _GJ_D     SAQWY         = TAQWY*KAQWY*YSDA ;

FRML _GJ_D     SARTPT        = TSARTPT*Y_S;
FRML _GJ_D     SARR          = TSARR*Y_S;
FRML _D        SAR           = SARTPT + SARR;
FRML _GJ_D     SAK           = TSAK*Y_S ;
FRML _D        SA            = SAK + SASO ;

// ERVHERVSFORDELTE SKATTER

FRML _D        SIPVEA        = (1-DXSIP)*TPVEA*FVEA+DXSIP*(PNCP/PNCP[-1])*TPVEA*FVEA ;
FRML _D        SIPVMA        = TPVMA*FVMA ;
FRML _D        SIPVA         = SIPVEA + SIPVMA + ((SIPAA+SIPSUER)-SIPEQ) ;

FRML _D        SIPVEB        = (1-DXSIP)*TPVEB*FVEB+DXSIP*(PNCP/PNCP[-1])*TPVEB*FVEB ;
FRML _D        SIPVMB        = TPVMB*FVMB ;
FRML _D        SIPVB         = SIPVEB + SIPVMB ;

FRML _D        SIPVEE        = (1-DXSIP)*TPVEE*FVEE+DXSIP*(PNCP/PNCP[-1])*TPVEE*FVEE ;
FRML _D        SIPVME        = TPVME*FVME ;
FRML _D        SIPVE         = SIPVEE + SIPVME ;

FRML _D        SIPVMN        = (1-DXSIP)*TPVMN*FVMN+DXSIP*(PNCP/PNCP[-1])*TPVMN*FVMN ;
FRML _D        SIPVN         = SIPVMN ;
FRML _D        SIPVMH        = (1-DXSIP)*TPVMH*FVMH+DXSIP*(PNCP/PNCP[-1])*TPVMH*FVMH ;
FRML _D        SIPVH         = SIPVMH ;

FRML _D        SIPVEI        = (1-DXSIP)*TPVEI*FVEI+DXSIP*(PNCP/PNCP[-1])*TPVEI*FVEI ;
FRML _D        SIPVMI        = TPVMI*FVMI ;
FRML _D        SIPVI         = SIPVEI + SIPVMI ;

FRML _D        SIPVET        = (1-DXSIP)*TPVET*FVET+DXSIP*(PNCP/PNCP[-1])*TPVET*FVET ;
FRML _D        SIPVMT        = TPVMT*FVMT ;
FRML _D        SIPVT         = SIPVET + SIPVMT ;

FRML _D        SIPVES        = (1-DXSIP)*TPVES*FVES+DXSIP*(PNCP/PNCP[-1])*TPVES*FVES ;
FRML _D        SIPVMS        = TPVMS*FVMS ;
FRML _D        SIPVS         = SIPVES + SIPVMS ;

FRML _D        SIPVEO        = (1-DXSIP)*TPVEO*FVEO+DXSIP*(PNCP/PNCP[-1])*TPVEO*FVEO ;
FRML _D        SIPVMO        = TPVMO*FVMO ;
FRML _D        SIPVO         = SIPVEO + SIPVMO ;

FRML _D        SIPV          = SIPVA + SIPVH + SIPVB + SIPVE + SIPVN + SIPVI + SIPVT + SIPVS + SIPVO ;

FRML _D        SIGVA         = BTGVA*TG*VA/(1+BTGVA*TG) ;
FRML _D        SIGVB         = BTGVB*TG*VB/(1+BTGVB*TG) ;
FRML _D        SIGVE         = BTGVE*TG*VE/(1+BTGVE*TG) ;
FRML _D        SIGVH         = BTGVH*TG*VH/(1+BTGVH*TG) ;
FRML _D        SIGVI         = BTGVI*TG*VI/(1+BTGVI*TG) ;
FRML _D        SIGVN         = BTGVN*TG*VN/(1+BTGVN*TG) ;
FRML _D        SIGVS         = BTGVS*TG*VS/(1+BTGVS*TG) ;
FRML _D        SIGVT         = BTGVT*TG*VT/(1+BTGVT*TG) ;
FRML _D        SIGVO         = BTGVO*TG*VO/(1+BTGVO*TG) ;

FRML _D        SIGV          = SIGVA + SIGVH + SIGVB + SIGVE + SIGVI + SIGVN + SIGVT + SIGVS + SIGVO ;

FRML _D        Siqw          = Siqu+Siqab+Siqam+Siqlt ;
FRML _DJ_      Siqwa         = bsiqua*Siqu + bsiqaba*Siqab + bsiqama*Siqam + bsiqlta*Siqlt ;
FRML _DJ_      Siqwb         = bsiqub*Siqu + bsiqabb*Siqab + bsiqamb*Siqam + bsiqltb*Siqlt ;
FRML _DJ_      Siqwn         = bsiqun*Siqu + bsiqabn*Siqab + bsiqamn*Siqam + bsiqltn*Siqlt ;
FRML _DJ_      Siqwe         = bsique*Siqu + bsiqabe*Siqab + bsiqame*Siqam + bsiqlte*Siqlt ;
FRML _DJ_      Siqwh         = bsiquh*Siqu + bsiqabh*Siqab + bsiqamh*Siqam + bsiqlth*Siqlt ;
FRML _DJ_      Siqwi         = bsiqui*Siqu + bsiqabi*Siqab + bsiqami*Siqam + bsiqlti*Siqlt ;
FRML _DJ_      Siqws         = bsiqus*Siqu + bsiqabs*Siqab + bsiqams*Siqam + bsiqlts*Siqlt ;
FRML _DJ_      Siqwo         = bsiquo*Siqu + bsiqabo*Siqab + bsiqamo*Siqam + bsiqlto*Siqlt ;
FRML _I        Siqwt         = Siqw-(Siqwa+Siqwb+Siqwn+Siqwe+Siqwh+Siqwi+Siqws+Siqwo) ;

FRML _GJ_D     Siqejh        = fKbh[-2]/(fKb[-2]-fKbo[-2])*Siqej ;
FRML _D        Siqejxh       = Siqej-Siqejh ;
FRML _GJ_      Siqa          = Siqwa+bsiqeja*Siqejxh+bsiqva*Siqv+bsiqco2a*Siqco2+bsiqra*Siqr+bsiqsqa*Siqsq+Siqaa  ;
FRML _GJ_      Siqb          = Siqwb+bsiqejb*Siqejxh+bsiqvb*Siqv+bsiqco2b*Siqco2+bsiqrb*Siqr+bsiqsqb*Siqsq        ;
FRML _GJ_      Siqn          = Siqwn+bsiqejn*Siqejxh+bsiqvn*Siqv+bsiqco2n*Siqco2+bsiqrn*Siqr+bsiqsqn*Siqsq        ;
FRML _GJ_      Siqe          = Siqwe+bsiqeje*Siqejxh+bsiqve*Siqv+bsiqco2e*Siqco2+bsiqre*Siqr+bsiqsqe*Siqsq+Siqpso ;
FRML _GJ_      Siqh          = Siqwh+bsiqejh*Siqejxh+bsiqvh*Siqv+bsiqco2h*Siqco2+bsiqrh*Siqr+bsiqsqh*Siqsq+Siqejh ;
FRML _GJ_      Siqi          = Siqwi+bsiqeji*Siqejxh+bsiqvi*Siqv+bsiqco2i*Siqco2+bsiqri*Siqr+bsiqsqi*Siqsq        ;
FRML _GJ_      Siqs          = Siqws+bsiqejs*Siqejxh+bsiqvs*Siqv+bsiqco2s*Siqco2+bsiqrs*Siqr+bsiqsqs*Siqsq        ;
FRML _GJ_      Siqo          = Siqwo+bsiqejo*Siqejxh+bsiqvo*Siqv+bsiqco2o*Siqco2+bsiqro*Siqr+bsiqsqo*Siqsq        ;
FRML _I        Siqt          = Siq-(Siqa+Siqb+Siqn+Siqe+Siqh+Siqi+Siqs+Siqo) ;

// Øvrige offentlige indtægter
FRML _GJ       TYPRI         = TTYPRI*Y_S ;

FRML _D        TAPO          = TAPOK + TAPOR ;
FRML _GJ_D     TAPOR         = TTAPOR*Y_S ;
FRML _GJ_D     TAPOK         = TKS*BTAPOK*YS ;

FRML _GJ_D     TKPO          = TTKPO*Y_S ;

FRML _D        TBPHO         = TPAF + TPEF + TPR ;
FRML _GJ_D     TPR           = TTPR*Y_S ;
FRML _D        TPAF          = TTPAF*PSRTY*UA*0.001 ;
FRML _D        TPEF          = TTPEF*PSRTY*BTPEF*UA*0.001 ;

FRML _GJ_D     TEUR          = TTEUR*Y_S ;

                                                     
// ************************************************  
// * 10.  BETALINGSBALANCE OG UDLANDSGÆLD         *  
// ************************************************  
FRML _D        TFFN          =  ENVT + TIFN + TWEN + TEUN
                                + TAFPN - TYPFU + (TAFO-TAOF) + TKFPN + (TKFO-TKOF) ;
FRML _D        ENL           =  TFFN - TKFPN - (TKFO-TKOF) + ENLR ;
FRML _D        ENVT          =  E - M ;

FRML _D        TEUN          =  TEUR - TEUBZ - SIM - SISUEU ;
FRML _GJ_D     TEUBZ         =  TTEUBZ*Y_S ;

FRML _D        SIQEU         =  SIQAA + SIQEUR ;

FRML _GJ_D     TAOFF         =  TTAOFF*Y_S ;
FRML _GJ_D     TAOFG         =  TTAOFG*Y_S ;
FRML _GJ_D     TAOFR         =  TTAOFR*Y_S ;
FRML _G        TAOF          =  TAOFF + TAOFG + TAOFR ;

FRML _GJ_D     TAFO          =  TTAFO*Y_S ;

FRML _GJ_D     TKOF          =  TTKOF*Y_S ;
FRML _GJ_D     TKFO          =  TTKFO*Y_S ;

FRML _GJ_D     TWEN          =  TTWEN*Y_S ;
FRML _GJ_D     TAFPN         =  TTAFPN*Y_S ;
FRML _GJ_D     TKFPN         =  TTKFPN*Y_S ;

FRML _G        WF            =  WF[-1] + TFFN + JWF ;
FRML _G        WFI           =  KWFI*Y_S ;
FRML _G        WFU           =  WFI-WF ;

FRML _G        TIFN          =  IWI*WFI[-1] - IWU*WFU[-1] + JTIFN ;

                                                     
// ************************************************  
// * 11. PRIVAT SEKTOR                            *  
// ************************************************  
// 11.1 INDKOMSTER
FRML _D        YD            = (YW-TYPRI) + (YR-IVOS) + TIPN + (TY-TYPFU) - SD;
FRML _D        YDNR          = YD + TWEN - SASO - TBPHO + TAOP - TAPO + TAFPN;
FRML _D        YDL_Y         = (YW-TYPRI) + (YR-YRO) + (TY-TYPFU) + TAFPN + TWEN + (TAOP-TAPOR) - (IV-IVOS*KIVO);      // YRO=KIVO*IVOS indsat /24.03.17 SAN
// YDL_S skal også korrigeres for SSYN, når skatter vedr. renter trækkes ud /29.10.18 DG
//GRML _D        YDL_S         = (SDK-(TSS0*KSSYS*KYS*TIPPPS)
//                               -(TSS0*KSSYS*KYS*KYSP+TSSP0*KSSYSP*KYSP)*(TPPLU-(TPPIL+TPPIK))) +
//                                SDV + SDU + SDA + SDPA + (SDS-TSDS*TIPPS) + SA + SDM + TBPHO + TAPOK;
FRML _D        YDL_S         = (SDK-(TSS0*KSSYS*KYS*TIPPPS-SSYN)
                               -(TSS0*KSSYS*KYS*KYSP+TSSP0*KSSYSP*KYSP)*(TPPLU-(TPPIL+TPPIK))) +
                                SDV + SDU + SDA + SDPA + (SDS-TSDS*TIPPS) + SA + SDM + TBPHO + TAPOK;
FRML _D        YDL           = YDL_Y - YDL_S;
FRML _D        YDK_Y         = (YW1-TYPRI) + (TY-TYPFU) + TIPPPS
                               + (TPPLU+KTPPKU*(TPPUK + TPPUA)-TPPI) + TWEN + YRH;
FRML _D        YDK_S         = SDK + SDV + SDU + SDA + SDP + SASO + SDM + TBPHO + TAPOK;
FRML _D        YDK           = YDK_Y - YDK_S;

FRML _D        FYDL          = YDL/PCP;
FRML _D        FYDK          = YDK/PCP;

FRML _D        TIPN          = TIFN - TION;
FRML _D        TIIPN         = TIFN - TIION;
FRML _D        TIPQN         = TIPN - TIPPN;
FRML _D        TIIPQN        = TIIPN - TIPPN;

// 11.2 FORBRUGSBESTEMMENDE FORMUE OG NETTOFORDRINGSERHVERVELSE
FRML _D        WCP           = KFKBHE*FKBH*PHK + (1-KFKBHE)*FKBH*PIBH + PCB*FKCB[-1]
                               + (WPQ[-1]+WPQ)/2
                               + ((1-TSDPK)*(WPPK[-1]+WPPK) + (1-(TSS0+TSSP0))*(WPPL[-1]+WPPL) + WPPA[-1]+WPPA)/2;
FRML _D        WP            = WF - WN_O;
FRML _D        WPQ           = WP - WPP;
FRML _D        TFPN          = TFFN - TFON;
FRML _D        TFPQN         = TFPN - TFPPN;

// 11.3 PENSIONER
// Pensionsindbetalinger
FRML _G        TPPIAF        = BTPPIAF*(1-tsda)*ysda;
FRML _G        TPPIAQ        = BTPPIAQ*(1-tsda)*ysda;
FRML _G        TPPIKF        = BTPPIKF*(1-tsda)*ysda;
FRML _G        TPPIKQ        = BTPPIKQ*(1-tsda)*ysda;
FRML _G        TPPILF        = BTPPILF*(1-tsda)*ysda  + TPPILFTY;  // Tilføjet pensionsindbetalinger fra indkomstoverførselsmodtagere 19.04.19 DG
FRML _G        TPPILQ        = BTPPILQ*(1-tsda)*ysda;

FRML _G        TPPILFTY      = BTPPILFTY*(TYD+TYKS+TYM+TYPE+TYPFO+TYUAK+TYUKAK+TYULY+TYURY);

FRML _G        TPPIA         = TPPIAF + TPPIAQ;
FRML _G        TPPIK         = TPPIKF + TPPIKQ;
FRML _G        TPPIL         = TPPILF + TPPILQ;

FRML _G        TPPI          = TPPIA + TPPIK + TPPIL;

FRML _G        TPPFI         = TPPIAF + TPPIKF + TPPILF;
FRML _G        TPPQI         = TPPIAQ + TPPIKQ + TPPILQ;

FRML _G        TPPUA         = BTPPUA*WPPA[-1];
FRML _G        TPPUK         = BTPPUK*WPPK[-1];
FRML _G        TPPUL         = BTPPUL*WPPL[-1];

FRML _G        NTPPIA        = (TIPPN + OPP - SDR)*(WPPA[-1]/WPP[-1]) + JNTPPIA;
FRML _G        NTPPIK        = (TIPPN + OPP - SDR)*(WPPK[-1]/WPP[-1]) + JNTPPIK;
FRML _G        NTPPIL        = (TIPPN + OPP - SDR)*(WPPL[-1]/WPP[-1]) - (JNTPPIA+JNTPPIK);

FRML _D        WPPA          = WPPA[-1] + NTPPIA + TPPIA - TPPUA;
FRML _D        WPPK          = WPPK[-1] + NTPPIK + TPPIK - TPPUK;
FRML _D        WPPL          = WPPL[-1] + NTPPIL + TPPIL - TPPUL;

FRML _D        WPP           = WPPA + WPPK + WPPL;

FRML _D        TPPU          = TPPUA + TPPUK + TPPUL;
FRML _D        TPPLU         = TPPUL;
FRML _D        TPPKU         = TPPUK;

FRML _D        BTPPFI        = TPPFI/((1-tsda)*ysda);

// Opbygning af pensionsformue
FRML _G        TIPPN         =  IWPP*WPP[-1];
FRML _D        TFPPN         =  TPPI - TPPU + TIPPN - SDR;
// JLED i OPP ændret fra JR til J, da OPP svinger vildt med både positive og negative tal, derfor svært at tilpasse med JR-led /29.10.18 DG
FRML _GJ_D     OPP           = Wpp[-1]*(0.5*kwpp + 0.5*(0.15+0.85*((1+iwbz)/(1+iwbz[-1]))**(-10) -1 )) ;
FRML _GJ_D     SAP           =  SAP[-1] + (BTPPFI - BTPPFI[-1]);

                                                     
// ************************************************  
// * 12. LØN                                      *  
// ************************************************  
//     12.1 LØN
FRML _SJRD     LNAP          = LNAP[-1]*EXP((1-0.375462)*(DLOG(PYFPBE)+ DLOG(VYFHPB_S))
                                             + 0.375462 * DLOG(LNAP[-1])
                                             + 0.482800 *(DLOG(PYFPB)-DLOG(PYFPBE))
                                             - 0.605341*0.5*Diff(BUL-BUL_S)
                                             + GLNAP
                                             - 0.605341*(BUL[-1] - BUL_S[-1])
                                             - 0.275272*(BYW1PB[-1] - BYW1PB_S)) ;
//GRML _DJRD     LOHKKW        = KLOHKKW*LNAP[-1] ;
//GRML _G        LOHKK         = (1-DXLOHKK)
//                                 *( LOHKK[-1]*EXP( DLOG(LNAP[-1]) -0.25*LOG(LOHKK[-1]/LOHKKW[-1]) )*(1+JRLOHKK) )
//                               + DXLOHKK*LOHKKX ;
// Tilbage til mere simpel ligning for off.løn /29.10.18 DG
FRML _GJRD     Dlog(LOHKK)   = Dlog(LNAP);

// BTYD udgår, da den alligevel kun er tabelvariabel og vi ikke ønsker at stå på mål for ADAM's historiske tal /26.02.19 DG
//_RML _GJ       BTYD          = PSRTY*TTYDD/(LNAP*HA*(1-TSDA)) ;

//     12.2 LØNSUMMER OG LØNKVOTER
FRML _G        YWA           = KLA*(LNAP*HGA*QWA*0.001) ;
FRML _G        YWB           = KLB*(LNAP*HGB*QWB*0.001) ;
FRML _G        YWE           = KLE*(LNAP*HGE*QWE*0.001) ;
FRML _G        YWH           = KLH*(LNAP*HGH*QWH*0.001) ;
FRML _G        YWN           = KLN*(LNAP*HGN*QWN*0.001) ;
FRML _G        YWI           = KLI*(LNAP*HGI*QWI*0.001) ;
FRML _G        YWT           = KLT*(LNAP*HGT*QWT*0.001) ;
FRML _G        YWS           = KLS*(LNAP*HGS*QWS*0.001) ;
FRML _G        YWO           = LOHKK*HGO*QWO*.001 ;
FRML _G        YW            = YWA+YWH+YWB+YWE+YWI+YWN+YWT+YWS+YWO ;

FRML _D        BYW           = YW/YF ;
FRML _D        BYWPS         = (YW-YWOS)/(YF-YFOS) ;
FRML _D        BYWPB         = (YWI+YWB+YWT)/(YFI+YFB+YFT) ;

// med imputeret løn til selvstændige
FRML _G        YW1A          = KLA*LNAP*HQA ;
FRML _G        YW1B          = KLB*LNAP*HQB ;
FRML _G        YW1E          = KLE*LNAP*HQE ;
FRML _G        YW1H          = KLH*LNAP*HQH ;
FRML _G        YW1N          = KLN*LNAP*HQN ;
FRML _G        YW1I          = KLI*LNAP*HQI ;
FRML _G        YW1T          = KLT*LNAP*HQT ;
FRML _G        YW1S          = KLS*LNAP*HQS ;

FRML _G        YW1           = YW1A+YW1H+YW1B+YW1E+YW1I+YW1N+YW1T+YW1S+YWO ;

FRML _G        BYW1          = YW1/YF ;
FRML _G        BYW1P         = (YW1-YWO)/(YF-YFO) ;
FRML _G        BYW1PS        = (YW1-YWOS)/(YF-YFOS) ;
FRML _G        BYW1PB        = (YW1I+YW1B+YW1T)/(YFI+YFB+YFT) ;

// 12.3 IMPLICIT TIMELØN
FRML _D        LA            = 1000*(YWA+SIQWA)/(QWA*HGA) ;
FRML _D        LB            = 1000*(YWB+SIQWB)/(QWB*HGB) ;
FRML _D        LE            = 1000*(YWE+SIQWE)/(QWE*HGE) ;
FRML _D        LN            = 1000*(YWN+SIQWN)/(QWN*HGN) ;
FRML _D        LH            = 1000*(YWH+SIQWH)/(QWH*HGH) ;
FRML _D        LI            = 1000*(YWI+SIQWI)/(QWI*HGI) ;
FRML _D        LT            = 1000*(YWT+SIQWT)/(QWT*HGT) ;
FRML _D        LS            = 1000*(YWS+SIQWS)/(QWS*HGS) ;
FRML _D        LO            = 1000*(YWO+SIQWO)/(QWO*HGO) ;

                                                     
// ************************************************  
// * 13. DEFLATORER                               *  
// ************************************************  
// 13.1 BVT-DEFLATORER

// PRIS PÅ BVT I B-ERHVERV
FRML _DJRD     PYFBW         =   (1/DTAB)*((LB/DTALFAB)**DTALFAB)*((PKZB/(1-DTALFAB))**(1-DTALFAB))
                               + (SIQB-SIQWB)/FYFB ;
FRML _D        VLBNP         = Dlog(LB)-(VTFPB/DTALFAB/100) ;
FRML _SJRD     Dlog(pyfb)    =   0.93744*vlbnp
                               - 0.25472*(Log(pyfb[-1])-Log(pyfbw[-1]))
                               + gpyfb ;

// PRIS PÅ BVT I I-ERHVERV
FRML _DJRD     PYFIW         =   (1/DTAI)*((LI/DTALFAI)**DTALFAI)*((PKZI/(1-DTALFAI))**(1-DTALFAI))
                               + (SIQI-SIQWI)/FYFI ;
FRML _D        VLINP         = Dlog(LI)-(VTFPI/DTALFAI/100) ;
FRML _SJRD     Dlog(pyfi)    =   0.5414*vlinp + 0.17302*vlinp[-1]
                               - 0.15*(Log(pyfi[-1])-Log(pyfiw[-1]))
                               + gpyfi ;

// PRIS PÅ BVT I T-ERHVERV
FRML _DJRD     PYFTW         =   (1/DTAT)*((LT/DTALFAT)**DTALFAT)*((PKZT/(1-DTALFAT))**(1-DTALFAT))
                               + (SIQT-SIQWT)/FYFT;
FRML _D        VLTNP         = Dlog(LT)-(VTFPT/DTALFAT/100) ;
FRML _SJRD     Dlog(pyft)    =   0.57559*vltnp + 0.25145*vltnp[-1]
                               - 0.1224*(Log(pyft[-1])-Log(pyftw[-1]))
                               + gpyft ;

// PRIS PÅ BVT I H-ERHVERV
FRML _G        PYFH          = Exp(Log(PIB) - 0.1 + jpyfh)+SIQH/FYFH ;

// PRIS PÅ BVT I ØVRIGE PRIVATE ERHVERV (A,N,S)
FRML _G        PYFA          = PYFAX - (SIPAA-SIPAAX+(SIPSUER-SIPSUERX))/FYFA ;
FRML _D        PYFN          = YFN/FYFN ;
FRML _D        PYFS          = YFS/FYFS ;

// 12.3 PRODUKTIONSVÆRDI-DEFLATORER
FRML _D        PXA           = XA/FXA ;
FRML _D        PXB           = XB/FXB ;
FRML _D        PXN           = (PEE/KPEE - (AEEE*PXE + AMEEE*PME + ASVEE*PSIV))/ANEE ;
FRML _D        PXE           = XE/FXE ;
FRML _D        PXH           = XH/FXH ;
FRML _D        PXI           = XI/FXI ;
FRML _D        PXT           = XT/FXT ;
FRML _D        PXS           = PMSS ;


// 12.3 NETTO-PRISER PÅ ENDELIG ANVENDELSE

// KORREKTIONSFAKTOR TIL PRISSAMMENBINDINGSRELATIONER
FRML _G        KKP           = KKP + 1 - (CP + CO + I + E - M - SIV)
                                 /(YFA + YFH + YFB + YFN + YFE + YFI + YFT + YFS + YFO) ;

// NETTOPRISER
FRML _G        PNCO          = (ATCO*PXT + AOCO*PXO)*KPNCO ;
FRML _G        PNCE          = (AECE*PXE + ATCE*PXT)*KPNCE*KKP ;
FRML _G        PNCG          = (AECG*PXE + ATCG*PXT + AMECG*(PME+TME))*KPNCG*KKP ;
FRML _G        PNCH          = (AHCH*PXH + ATCH*PXT)*KPNCH*KKP ;
FRML _G        PNCV          = (AACV*PXA + AICV*PXI + ATCV*PXT + AMICV*(PMI+TMI))*KPNCV*KKP ;
FRML _G        PNCS          = (ATCS*PXT + AOCS*PXO + AMSQCS*PMSQ)*KPNCS*KKP ;
FRML _G        PNCB          = (ATCB*PXT + AMICB*(PMI+TMI))*KPNCB*KKP ;
FRML _G        PNIMX         = (AIIM*PXI + ATIM*PXT + AMIIM*(PMI+TMI) + AMSQIM*PMSQ)*KPNIMX ;
FRML _G        PNIB          = (ABIB*PXB + ATIB*PXT)*KPNIB ;
FRML _G        PNEI          = (AAEI*PXA + AIEI*PXI + ATEI*PXT + AMIEI*(PMI+TMI))*KPNEI ;
// KP-led fjernet i nettoprisindeks, der i stedet defineres ud fra forbrugsudgift regnet i nettopriser /26.02.19 DG
//_RML _G        PNCP          = ((FCE*PNCE+FCV*PNCV+PNCS*FCS+PCT*FCT+PNCB*FCB+
//                                 PNCG*FCG+PNCH*FCH-PET*FET)/FCP)*KPNCP ;
FRML _D        PNCP          = ((FCE*PNCE+FCV*PNCV+PNCS*FCS+PCT*FCT+PNCB*FCB+
                                 PNCG*FCG+PNCH*FCH-PET*FET)/FCP) ;

FRML _G        PIL           = (0.1*PXE + 0.3*PXI + 0.2*PXT + 0.1*PME + 0.3*PMI)*KPIL*KKP ;

// 13.4 MARKEDS-PRISER PÅ ENDELIGE ANVENDELSER
FRML _D        PCO           = (1+BTGCO*TG)*PNCO*(1+TRCO) ;
FRML _D        PCE           = (1+BTGCE*TG)*(PNCE+TPCE) ;
FRML _D        PCG           = (1+BTGCG*TG)*(PNCG+TPCG) ;
FRML _D        PCS           = (1+BTGCS*TG)*(PNCS+TPCS+SIPUR/FCS) ;
FRML _D        PCT           = PMT ;
FRML _D        PCV           = (1+BTGCV*TG)*(PNCV+TPCV) ;
FRML _D        PCH           = (1+BTGCH*TG)*(PNCH+TPCH) ;
FRML _D        PCB           = (1+BTGCB*TG)*(PNCB+TPCB)*(1+TRCB) ;

FRML _D        PIBPS         = (1+BTGIBPS*TG)*(KPIBPS*PNIB+TPIBPS) ;
FRML _D        PIBOS         = (1+BTGIBOS*TG)*(KPIBOS*PNIB+TPIBOS) ;
FRML _D        PIBH          = (1+BTGIBH*TG)*(KPIBH*PNIB+TPIBH) ;

FRML _D        PIMPS         = (1+BTGIMPS*TG)*(KPIMPS*PNIMX+TPIMPS)*(1+TRIMPS) ;
FRML _D        PIMXOS        = (1+BTGIMOS*TG)*(KPIMXOS*PNIMX+TPIMOS) ;

FRML _D        PIV           = KPIV*(0.50*PIM + 0.50*PIB) ;
FRML _D        PIVOS         = KPIVOS*(0.5*PIMOS + 0.5*PIBOS) ;

FRML _D        PEI           = PNEI + SIPEI/FEI ;
FRML _D        PESQ          = (ABESQ*PXB + AIESQ*PXI + ATESQ*PXT + AMSQESQ*PMSQ)*KPESQ ;
FRML _D        PESS          = PXS ;
FRML _D        PET           = (0.1*PCE + 0.4*PCV + 0.5*PCS)*KPET ;


// 13.5 ENERGIPRISER
// DEFLATOR PÅ ENERGIIMPORT
FRML _DJRD     Log(pmew)     = Log(pwoil*vusa) + kpmew ;
FRML _SJRD     Dlog(pme)     = 0.77758*Dlog(pwoil*vusa) + gpme
                               -0.1998*Log(pme[-1]/pmew[-1]) ;

// DEFLATOR PÅ ENERGIEKSPORT
FRML _DJRD     Log(peew)     = Log(pwoil*vusa) + kpeew ;
FRML _SJRD     Dlog(pee)     = 0.81594*Dlog(pwoil*vusa) + gpee
                               -0.30597*Log(pee[-1]/peew[-1]) ;

// 13.6 DEFLATORER PÅ AGGREGATER
FRML _D        PY            = 1*Y/FY ;
FRML _D        PYF           = YF/FYF ;
FRML _D        PYFP          = YFP/FYFP ;
FRML _D        PYFPB         = (YFI + YFB + YFT)/FYFPB ;
FRML _D        PYFO          = YFO/FYFO ;
FRML _D        PYFR          = YFR/FYFR ;

FRML _D        PX            = X/FX ;
FRML _D        PXO           = XO/FXO ;

FRML _D        PVX           = 1*VX/FVX ;
FRML _D        PSIV          = SIV/FSIV ;

FRML _D        PCP           = CP/FCP ;

FRML _D        PIMOS         = IMOS/FIMOS ;
FRML _D        PIXOS         = IXOS/FIXOS ;
FRML _D        PIOS          = IOS/FIOS ;

FRML _D        PIB           = (PIBPS*FIBPS + PIBH*FIBH + PIBOS*FIBOS)/FIB ;
FRML _D        PIMX          = IMX/FIMX ;
FRML _D        PIM           = IM/FIM ;
FRML _D        PIF           = IF/FIF ;
FRML _D        PI            = I/FI ;

FRML _D        PAIPS         = AIPS/FAIPS ;
FRML _D        PAI           = AI/FAI ;

FRML _D        PEV           = EV/FEV ;
FRML _D        PES           = ES/FES ;
FRML _D        PEST          = EST/FEST ;
FRML _D        PE            = E/FE ;

FRML _D        PAE           = AE/FAE ;

FRML _D        PM            = M/FM ;
FRML _D        PMV           = MV/FMV ;
FRML _D        PMS           = MS/FMS ;
FRML _D        PMST          = MST/FMST ;

FRML _D        PAT           = AT/FAT ;

                                                     
// ************************************************  
// * 14. VÆRDIER                                  *  
// ************************************************  
// 14.1 TILGANG
// ERHVERVSFORDELT PRODUKTION I ÅRETS PRISER
FRML _D        XA            = YFA + VA ;
FRML _D        XB            = YFB + VB ;
FRML _D        XN            = FXN*PXN ;
FRML _D        XE            = YFE + VE ;
FRML _D        XH            = YFH + VH ;
FRML _D        XI            = YFI + VI ;
FRML _D        XT            = YFT + VT ;
FRML _D        XS            = FXS*PXS ;

// ERVHERVSFORDELT VAREFORBRUG I ÅRETS PRISER
FRML _G        VA            = FXA*(AEA*PXE +
                               AAA*PXA + AIA*PXI + ATA*PXT + AMIA*(PMI+TMI))*KPVA
                               + SIPVA + SIGVA ;
FRML _G        VB            = FXB*(AEB*PXE +
                               AIB*PXI + ATB*PXT + AMIB*(PMI+TMI) + AMSQB*PMSQ)*KPVB
                               + SIPVB + SIGVB ;
FRML _G        VN            = FXN*(ATN*PXT)*KPVN
                               + SIPVN + SIGVN ;
FRML _G        VE            = FXE*(ANE*PXN + AEE*PXE + AMEE*(PME+TME) +
                               ATE*PXT)*KPVE
                               + SIPVE + SIGVE ;
FRML _G        VH            = FXH*(ABH*PXB + ATH*PXT)*KPVH
                               + SIPVH + SIGVH ;
FRML _G        VI            = FXI*(AEI*PXE +
                               AAI*PXA + AII*PXI + ATI*PXT + AMII*(PMI+TMI) + AMSQI*PMSQ)*KPVI
                               + SIPVI + SIGVI ;
FRML _G        VT            = FXT*(AET*PXE + AMET*(PME+TME) +
                               ABT*PXB + AIT*PXI + ATT*PXT + AST*PXS + AMIT*(PMI+TMI) + AMSQT*PMSQ)*KPVT
                               + SIPVT + SIGVT ;
FRML _G        VS            = FXS*(AMES*(PME+TME) +
                               ATS*PXT + AMSSS*PMSS)*KPVS
                               + SIPVS + SIGVS ;

// ERHVERVSFORDELT BVT I ÅRETS PRISER
FRML _D        YFA           = PYFA*FYFA ;
FRML _D        YFB           = PYFB*FYFB ;
FRML _D        YFN           = XN - VN ;
FRML _D        YFE           = PYFE*FYFE ;
FRML _D        YFH           = PYFH*FYFH ;
FRML _D        YFI           = PYFI*FYFI ;
FRML _D        YFT           = PYFT*FYFT ;
FRML _D        YFS           = XS - VS ;

// ERHVERVSFORDELT RESTINDKOMST
FRML _D        YRI           = YFI - YWI - SIQI ;
FRML _D        YRT           = YFT - YWT - SIQT ;
FRML _D        YRE           = YFE - YWE - SIQE ;
FRML _D        YRN           = YFN - YWN - SIQN ;
FRML _D        YRB           = YFB - YWB - SIQB ;
FRML _D        YRH           = YFH - YWH - SIQH ;
FRML _D        YRA           = YFA - YWA - SIQA ;
FRML _D        YRS           = YFS - YWS - SIQS ;
FRML _D        YRO           = YFO - YWO - SIQO ;

// 14.2 BNP, BVT, FORBRUG, INVESTERINGER, IMPORT OG EKSPORT
FRML _D        Y             = CP + CO + I - M + E ;
FRML _D        YR            = YF - YW - SIQ ;

FRML _D        YFPB          = YFB+YFI+YFT ;
FRML _D        YFPR          = YFA+YFE+YFN+YFH+YFS ;
FRML _D        YFR           = YFPR + YFO ;
FRML _D        YFP           = YFPB+YFPR ;
FRML _D        YF            = YFP + YFO ;

FRML _D        XPB           = XB+XI+XT ;
FRML _D        XPR           = XA+XN+XE+XH+XS ;
FRML _D        XP            = XPB+XPR ;

FRML _D        VX            = (VA + VB + VH + VI + VE + VN + VT + VS + VO) ;

FRML _D        AT            = YF + M + SIV ;

FRML _D        X             = YF + VX ;

FRML _D        CA            = FCE*PCE + FCG*PCG + FCV*PCV + FCS*PCS + FCT*PCT + FCH*PCH - FET*PET ;
FRML _D        CB            = FCB*PCB ;
FRML _D        CP            = FCA*PCA + FCB*PCB ;

FRML _D        IB            = PIB*FIB ;
FRML _D        IBOS          = PIBOS*FIBOS ;
FRML _D        IMX           = IMPS + IMXOS ;
FRML _D        IM            = IMX + IOFU ;
FRML _D        IMXOS         = PIMXOS*FIMXOS ;
FRML _D        IMOS          = IMXOS + IOFU ;
FRML _D        IXOS          = IBOS + IMXOS ;

FRML _D        IOS           = IMOS + IBOS ;
FRML _D        IMPS          = FIMPS*PIMPS ;
FRML _D        IBPS          = FIBPS*PIBPS ;
FRML _D        IBH           = FIBH*PIBH ;
FRML _D        IF            = I - IL ;
FRML _D        IV            = PIV*FIV ;
FRML _D        I             = PIBH*FIBH +
                               PIMPS*FIMPS + PIMXOS*FIMXOS + PIOFU*FIOFU +
                               PIBOS*FIBOS + PIBPS*FIBPS +
                               PIL*FIL ;

FRML _D        ME            = PME*FME ;
FRML _D        MI            = PMI*FMI ;
FRML _D        MV            = ME + MI ;
FRML _D        MT            = PMT*FMT ;
FRML _D        MSS           = PMSS*FMSS ;
FRML _D        MSQ           = PMSQ*FMSQ ;
FRML _D        MS            = MSS + MSQ ;
FRML _D        MST           = MS + MT ;
FRML _D        M             = MV + MST ;

FRML _D        EE            = PEE*FEE ;
FRML _D        EI            = PEI*FEI ;
FRML _D        EV            = EE + EI ;
FRML _D        ESS           = PESS*FESS ;
FRML _D        ESQ           = PESQ*FESQ ;
FRML _D        ET            = PET*FET ;
FRML _D        ES            = ESS + ESQ ;
FRML _D        EST           = ES + ET ;
FRML _D        E             = EV + EST ;

FRML _D        AI            = CP+CO+I ;
FRML _D        AIPS          = AI-CO-IOS ;
FRML _D        AE            = CP+CO+I+E ;

FRML _D        YI            = TWEN + TIFN - (SIPEE + SIPAA + SIPSUER + SIQEU) - (SIM + SIPEU) + Y ;

                                                     
// ************************************************  
// * 15. RENTER OG VALUTAKURS                     *  
// ************************************************  
FRML _GJ_D     IWBZ          = 0.2*IWBECB + 0.35*IWB10YS + 0*IWB10YSDEU +0.45*IWB30YR + RPIWBZ ;
FRML _GJ_D     IWLO          = 0.3*IWBECB + 0.7 *IWB10YS + 0*IWB10YSDEU +   0*IWB30YR + RPIWLO ;
FRML _GJ_D     IWDEH         =     IWBECB                                             + RPIWDEH ;

FRML _GJ_D     IWI           = 0.5*IWI[-1]  + 0.5*(0.5*IWBECB + 0.5*IWB10YSDEU + RPIWI) ;
FRML _GJ_D     IWU           = 0.5*IWU[-1]  + 0.5*(0.5*IWBECB + 0.5*IWB10YS + RPIWU) ;
FRML _GJ_D     IWPP          = 0.5*IWPP[-1] + 0.5*(IWB30YR + RPIWPP) ;

FRML _GJRD     EFFKR         = Exp(- 0.089*Dlog(VGBR) - 0.059*Dlog(VJPN) - 0.172*Dlog(VUSA)
                                   - 0.065*Dlog(VNOR) - 0.104*Dlog(VSWE) - 0.511*Dlog(VDEU))*EFFKR[-1] ;

                                                     
// ************************************************  
// * 16. STRUKTURELLE NIVEAUER OG GAPS            *  
// ************************************************  
FRML _D        Q_S           = UA_S - UL_S ;
FRML _D        HQ_S          = (HG_S/1000)*Q_S ;
FRML _D        HQR_S         = DXHQR_S*HQR + (1-DXHQR_S)*HQR_SX ;
FRML _D        HQPB_S        = HQ_S-HQR_S ;

FRML _D        KHGW          = DXHG*((((HQA/KHGA)+(HQB/KHGB)+(HQI/KHGI)+(HQT/KHGT)+(HQN/KHGN)+(HQE/KHGE)+(HQS/KHGS)+(HQH/KHGH)+(HQO/KHGO))
                                 /HQ)*HG_S/HA)
                               + (1-DXHG)*KHGWX ;

FRML _D        VYFHR_S       = DXVYFHR_S*VYFHR + (1-DXVYFHR_S)*VYFHR_SX ;
FRML _D        BPB_S         = DXBPB_S*((HQPB_S/HQ_S)*(PYFPB[-1]/PYF[-1])) + (1-DXBPB_S)*BPB_SX ;
FRML _D        BR_S          = DXBR_S*((HQR_S/HQ_S)*(PYFR[-1]/PYF[-1])) + (1-DXBR_S)*BR_SX ;
FRML _D        VYFHPB_S      = Exp(LOGTFP_S+(1-BYW1PB_S)*Log(FKPB/HQPB_S)) ;
FRML _D        TFPPB         = Exp(Log(FYFPB)-(1-BYW1PB_S)*Log(FKPB)-BYW1PB_S*Log(HQPB)) ;
FRML _D        VYFH_S        = BPB_S*VYFHPB_S + BR_S*VYFHR_S ;

FRML _D        AFGIFT_S      = DXAFGIFT_S*(FY/FYF) + (1-DXAFGIFT_S)*AFGIFT_SX ;

FRML _D        FYF_S         = VYFH_S*HQ_S ;
FRML _D        FY_S          = FYF_S*AFGIFT_S ;
FRML _D        PY_S          = DXPY_S*PY + (1-DXPY_S)*PY_SX ;
FRML _D        Y_S           = PY_S*FY_S ;

FRML _D        AFGIFT_GAP    = ((FY/FYF)-AFGIFT_S)/AFGIFT_S*100 ;
FRML _D        TFP_GAP       = (Log(TFPPB)-LOGTFP_S)*100 ;
FRML _D        KL_GAP        = (Log(FKPB/HQPB)-LOGKL_TR)*100 ;
FRML _D        VYFHPB_GAP    = BPB_S*(VYFHPB-VYFHPB_S)/VYFHPB_S*100 ;
FRML _D        VYFH_GAP      = (VYFH-VYFH_S)/VYFH_S*100 ;
FRML _D        HG_GAP        = (HG-HG_S)/HG_S*100 ;
FRML _D        UL_GAP        = ((UL/UA)-(UL_S/UA_S))*100 ;
FRML _D        UA_GAP        = (UA-UA_S)/UA_S*100 ;
FRML _D        Q_GAP         = (Q-Q_S)/Q_S*100 ;
FRML _D        HQ_GAP        = (HQ-HQ_S)/HQ_S*100 ;
FRML _D        FY_GAP        = (FY-FY_S)/FY_S*100 ;
FRML _D        FYF_GAP       = (FYF-FYF_S)/FYF_S*100 ;
FRML _D        Y_GAP         = (Y-Y_S)/Y_S*100 ;
FRML _D        PY_GAP        = (PY-PY_S)/PY_S*100 ;

FRML _D        VYFHPB_TR     = Exp(LOGTFP_S+(1-BYW1PB_S)*LOGKL_TR) ;
FRML _D        FY_TR         = AFGIFT_S*(BPB_S*VYFHPB_TR+BR_S*VYFHR_S)*(HG_S/1000)*(1-(UL_S/UA_S))*UA_S ;

                                                     
AFTER $                                              
                                                     
                                                     
// ************************************************  
// * TABELVARIABLER OG GENERERING AF JLED         *  
// ************************************************  
                                                     
                                                     
FRML YJNTPPIK  JNTPPIK       = NTPPIK - (TIPPN + OPP - SDR)*WPPK[-1]/WPP[-1];
FRML YJNTPPIA  JNTPPIA       = NTPPIA - (TIPPN + OPP - SDR)*WPPA[-1]/WPP[-1];
// GRML YJOPP     JOPP          = OPP - (WPP[-1] * (-0.4 * ((IWBZ/IWBZ[-1] - 1)+ (IWBZ[-1]/IWBZ[-2] - 1)) / 2 +  KWPP));

                                                     
FRML YJgpibe   Jgpibe        = gpibe - ( 0.25*gpib + 0.75*gpibe ) ;
FRML YZgpibe   Zgpibe        = gpibe ;

FRML YJgpime   Jgpime        = gpime - ( 0.25*gpim + 0.75*gpime ) ;
FRML YZgpime   Zgpime        = gpime ;
                                                     
FRML YJgpcpe   Jgpcpe        = gpcpe - ( 0.5*gpcpe[-1] + 0.5*(pcp/pcp[-1]-1) ) ;
FRML YZgpcpe   Zgpcpe        = gpcpe ;

                                                     
FRML YHQAX     HQAX          = HQA ;
FRML YVYFHAX   VYFHAX        = FYFA/HQA ;
FRML YVYFHBX   VYFHBX        = FYFB/HQB ;
FRML YVYFHIX   VYFHIX        = FYFI/HQI ;
FRML YVYFHTX   VYFHTX        = FYFT/HQT ;

FRML YJRHQB    JRhqb         = log(hqb) - (0.80208*Log(hqbn) + 0.19792*Log(hqbn[-1])) ;
FRML YJRHQI    JRhqi         = log(hqi) - (0.50458*log(hqin) + 0.49542*log(hqin[-1])) ;
FRML YJRHQT    JRhqt         = log(hqt) - (0.61527*log(hqtn) + 0.38473*log(hqtn[-1])) ;


                                                     
FRML YJPYFH    JPYFH         = Log(PYFH - SIQH/FYFH) - (Log(PIB)-0.1) ;

                                                     
FRML YJEFFKR   JEFFKR        = Log( EFFKR/EFFKR[-1]) - (-0.089*DLOG(VGBR)-0.059*DLOG(VJPN)- 0.172*DLOG(VUSA)
                                                        -0.065*DLOG(VNOR)-0.104*DLOG(VSWE)- 0.511*DLOG(VDEU));
                                                     
FRML YJPSRTY   JPSRTY        = PSRTY/PSRTY[-1]-(((LNAP[-2]*(1-SAP[-2])*HA[-2])/(LNAP[-3]*(1-SAP[-3])*HA[-3]))*(1-TSDA)/(1-TSDA[-1])) ;

FRML YJPTTYP   JPTTYP        = PTTYP/PTTYP[-1]-(PCP[-2]/PCP[-3]) ;

FRML YJIOFU    JIOFU         = IOFU - TIOFU*Y_S ;

FRML YKSSYEJ   KSSYEJ        = SSYEJ/(TQKEJ*PHV*(KFKBHE[-1]*FKBH[-1]/0.8)) ;

FRML YKSIQEJ   KSIQEJ        = SIQEJ/(TSIQEJ*PHV[-1]) ;

FRML YBIVOS    BIVOS         = IVOS/CO ;

FRML YKFIVOS   KFIVOS        = FIVOS/FIVO ; // Tilføjet baglænsberegning af k-faktor (var gjort direkte i SMEC.FRM til F19) /24.06.19 DG
