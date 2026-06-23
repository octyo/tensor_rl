# Køreplan: Implementér en CP-agent og sammenlign med en tabulær baseline

**Mål i én sætning:** Udskift `Q`-tabellen i tabulær Q-læring med en **lavrangs
CP-tensor**, og vis empirisk *hvornår* det giver samme politik med langt færre
parametre — og hvornår det ikke gør.

I implementerer alt selv (ingen færdige tensor-biblioteker til selve agenten —
`numpy` til array-regning er fint). Køreplanen giver jer formlerne og de få
fælder, der ellers koster en uge.

---

## 1. Hvilken algoritme: CP Q-læring (online, tabulær-agtig)

Vælg **denne ene** algoritme til kerneprojektet. Det er den simpleste CP-agent og
den, der er *mest direkte sammenlignelig* med en tabulær baseline: samme
trænings-loop, samme opdatering — den eneste forskel er, hvordan `Q` gemmes.
(SARSA, Expected SARSA, fitted-Q m.fl. er gode *udvidelser*, når dette virker —
ikke startpunktet.)

Opdateringen er almindelig semi-gradient Q-læring. For en transition
`(s, a, r, s')` med diskonteringsfaktor `γ`:

$$\delta = r + \gamma \max_{a'} Q[s',a'] - Q[s,a] \qquad\text{(TD-fejlen)},$$

hvorefter `Q[s,a]` skubbes mod sit mål. Hele forskningsspørgsmålet er: *hvad sker
der med læringen, når `Q` tvinges til at være lavrangs?*

### Den konkrete algoritme (samlet)

Notation: en tilstand er et faktor-indeks `s = (s_1,…,s_m)` og en handling
`a = (a_1,…,a_k)`; det fulde tensor-indeks for et `(s,a)`-par er sammenkædningen
`(s_1,…,s_m, a_1,…,a_k)`. `entry(A, idx)` og `entry_grad(A, idx)` er
hjælpefunktionerne fra afsnit 3. Den eneste forskel fra tabulær Q-læring er, at
`Q` *er* CP-faktormatricerne `A`, og at opdateringen er det normaliserede skridt.

```
ALGORITME 1 — CP Q-læring (ε-grådig, online)

Input:  miljø; rang R; læringsrate lr (≈ 0.3–0.5); diskontering γ;
        ε-skema (ε_start, ε_min, decay); antal episoder T; ε_0 = 1e-8
Output: faktormatricer A_1,…,A_N, der repræsenterer Q

1:  for hver akse d = 1..N:                       # N = #tilstandsfaktorer + #handlingsfaktorer
2:      A_d  ←  N(0, σ²)-matrix af form (aksestørrelse_d × R),  σ = R^(−1/(2N))
3:  ε ← ε_start
4:  for episode = 1..T:
5:      s ← miljø.reset()
6:      gentag indtil terminal:
7:          # ── ε-grådigt handlingsvalg ──
8:          med sandsynlighed ε:   a ← tilfældig handling
9:          ellers:                a ← argmax_{a'}  entry(A, (s, a'))   # over alle handlinger a'
10:         (s', r, terminal) ← miljø.step(a)
11:         # ── TD-mål ──
12:         hvis terminal:  y ← r
13:         ellers:         y ← r + γ · max_{a'} entry(A, (s', a'))
14:         # ── normaliseret (NLMS) semi-gradient opdatering af cellen (s,a) ──
15:         idx   ← (s_1,…,s_m, a_1,…,a_k)
16:         δ     ← y − entry(A, idx)
17:         g     ← entry_grad(A, idx)                 # liste med N gradient-rækker
18:         norm2 ← Σ_d ‖g_d‖²  +  ε_0
19:         for d = 1..N:   A_d[idx_d]  ←  A_d[idx_d] + lr · δ · g_d / norm2
20:         s ← s'
21:     ε ← max(ε_min, ε · decay)                      # udforskning aftager
```

Linje 9 og 13 er de eneste steder, handlingsrummet gennemløbes — derfor er det
billigt for små handlingsrum (én faktor). Den **tabulære baseline er nøjagtig
samme algoritme**, blot med `Q` som et tæt array og linje 14–19 erstattet af
`Q[s,a] ← Q[s,a] + lr·δ` (lille `lr`). Det gør de to direkte sammenlignelige.

---

## 2. Baggrund I skal bruge (kort)

`Q` er en opslagstabel: giv den en tilstand og en handling, få en værdi. I et
**faktoriseret** problem er det et flerdimensionelt gitter
`Q[s_1,…,s_m, a_1,…,a_k]` — én celle per kombination. Antallet af celler
**ganges** sammen på tværs af akser og eksploderer derfor.

En **CP-dekomposition** (CANDECOMP/PARAFAC) approksimerer hele gitteret som en
**sum af nogle få simple byggeklodser**. Hver byggeklods giver hver akse sit eget
lille "scorekort" (én værdi per indeks), og værdien i en celle er scorerne for den
celles koordinater **ganget sammen**. Antallet af byggeklodser kaldes **rangen** `R`.

For en tensor med `N` akser og rang `R`, repræsenteret ved én **faktormatrix**
`A_d` per akse (formen `(aksens størrelse, R)`), er en celle:

$$Q[i_1,\dots,i_N] = \sum_{r=1}^{R} \prod_{d=1}^{N} A_d[i_d,\, r].$$

Parameterantallet går fra `∏(aksestørrelser)` (tabel) til `R·Σ(aksestørrelser)`
(CP) — fra *multiplikativt* til *additivt*. Det er hele pointen.

---

## 3. Sådan implementerer I CP-agenten

### (a) Repræsentation
Gem én faktormatrix `A_d` per akse: én for hver tilstandsfaktor og én for hver
handlingsfaktor. `R` er en hyperparameter, I varierer senere.

### (b) Læs én celle (`entry`)
```python
def entry(factors, index):          # index = (i_1, ..., i_N)
    prod = np.ones(R)
    for d in range(N):
        prod = prod * factors[d][index[d]]   # elementvis produkt over rang-aksen
    return float(prod.sum())
```
Bemærk: I bygger **aldrig** hele tensoren — I tager én række fra hver faktormatrix,
ganger dem elementvist og summerer. Det er præcis derfor metoden passer til RL,
hvor I kun rører ét `Q[s,a]` per skridt.

### (c) Gradient mht. faktorrækkerne (`entry_grad`)
Gradienten af cellen mht. række `A_d[i_d, :]` er **produktet af alle de andre
aksers valgte rækker** ("leave-one-out"):
$$\frac{\partial Q[i]}{\partial A_d[i_d,:]} = \prod_{d' \neq d} A_{d'}[i_{d'},:].$$
```python
def entry_grad(factors, index):
    rows = [factors[d][index[d]] for d in range(N)]
    grads = []
    for d in range(N):
        loo = np.ones(R)
        for d2 in range(N):
            if d2 != d:
                loo = loo * rows[d2]
        grads.append(loo)           # gradient mht. række index[d] i faktor d
    return grads
```

### (d) Opdateringen — og den fælde, der betyder mest
Den naive semi-gradient-opdatering `A_d[i_d,:] += lr·δ·loo` **divergerer** på
tensorer med mange akser. Grunden: den faktiske ændring i `Q[s,a]` skalerer med
`‖gradient‖²`, som vokser med (antal akser · rang) — så skridtet skyder over.

Brug i stedet det **normaliserede (NLMS) skridt**: divider hele opdateringen med
`‖gradient‖² + ε`:
```python
def td_update(factors, sa_index, target, lr, eps=1e-8):
    delta = target - entry(factors, sa_index)
    grads = entry_grad(factors, sa_index)
    norm2 = sum(float((g*g).sum()) for g in grads) + eps
    for d in range(N):
        factors[d][sa_index[d]] += lr * delta * grads[d] / norm2
    return delta
```
Så bliver ændringen i `Q[s,a]` ≈ `lr·δ`, uanset antal akser — præcis som en
tabulær opdatering. Konsekvens: **`lr` er stor her, ca. 0.3–0.5**, *ikke* de små
værdier, tæt Q-læring bruger. Skriv det på tavlen; det er den hyppigste grund til,
at projektet går i stå.

> Hvorfor virker det: ændringen i cellen er
> `ΔQ ≈ Σ_d grad_d · Δrække_d = lr·δ·(Σ_d ‖grad_d‖²)/norm2 ≈ lr·δ`.

### (e) Initialisering (akse-bevidst)
Initialisér hver faktorindgang som `N(0, σ²)` med
$$\sigma = R^{-1/(2N)}.$$
Det gør de rekonstruerede celler enheds-varians, så gradienterne hverken forsvinder
eller eksploderer, når antallet af akser vokser. En fast lille initialisering
ødelægger stille og roligt læringen på mange-akse-tensorer.

### (f) Handlingsvalg (argmax)
For et lille handlingsrum (én handlingsfaktor): beregn `Q[s,·]` for alle handlinger
og tag argmax. Det er helt fint til kerneprojektet. (Den "faktoriserede argmax" er
kun nødvendig for store multi-agent-handlingsrum — lad det være en udvidelse.)
Tilføj `ε`-grådig udforskning som i tabulær Q-læring.

---

## 4. Den tabulære baseline

Identisk trænings-loop, identiske delte hyperparametre (`γ`, `ε`-skema). `Q` er et
tæt `(antal_tilstande, antal_handlinger)` array, og opdateringen er den sædvanlige
`Q[s,a] += lr·δ` med en normal (lille) `lr`. **Få denne til at virke først** — det
er jeres opvarmning og jeres referencepunkt.

---

## 5. Miljøer (vælg dem, så kontrasten bliver tydelig)

Hold dem små nok til, at I kan beregne det eksakte optimale `Q*` med
**værdiiteration** — det låser op for den bedste metrik (afsnit 6).

- **En faktoriseret gitterverden** (fx `Q[række, kolonne, handling]`): glat, svagt
  koblet → **lavrangs** `Q*`. Her *bør* CP vinde (matche tabellen med færre
  parametre). Det positive resultat.
- **(Valgfrit, til den ærlige kontrast) et sværere, koblet miljø**, hvor `Q*` er
  højrangs — her *bør* CP ikke kunne komprimere og kræve næsten fuld rang. Det gør
  konklusionen skarp: *CP hjælper netop, når `Q*` er lavrangs.*

I skal selv kunne sætte miljøets tilstand (til værdiiteration) og kende dynamik +
belønning. En deterministisk gitterverden med `-1` per skridt og terminal ved mål
er rigeligt.

---

## 6. Sådan sammenligner I — metrikker

Kør alt over **≥5 tilfældige seeds** og rapportér gennemsnit ± et bånd. Fire
metrikker, hver sin figur:

1. **Parameterantal** — CP `R·Σ(størrelser)` mod tabel `∏(størrelser)`. Hele pointen.
2. **Læringskurve** — afkast (samlet belønning per episode) mod episoder, glattet og
   midlet over seeds. Når CP samme niveau?
3. **Værdinøjagtighed** (den metrik, de fleste projekter *ikke* kan lave, men I kan,
   fordi I har `Q*`): den relative fejl `‖Q̂ − Q*‖ / ‖Q*‖` mellem det lærte `Q` og
   det sande `Q*`, plottet hen over træningen. Den fortæller, om CP lærte de
   *rigtige værdier* — ikke bare en habil politik. Brug Frobenius-normen (kvadratrod
   af summen af kvadrerede celleforskelle).
4. **Ydelse-mod-parametre (hovedfiguren)** — sweep CP-rangen `R = 1, 2, 4, 8, …` og
   plot slut-afkast (og slut-værdifejl) mod parameterantal, med tabellen som ét
   referencepunkt. Historien fremstår som en front: CP matcher tabellen, så snart
   `R` når værdifunktionens iboende rang — ved en brøkdel af parametrene.

> **Ekstra indsigt, hvis I har tid:** Find den mindste rang, hvor en *direkte*
> CP-tilpasning af `Q*` (ikke læring, bare mindste-kvadraters tilpasning af tensoren)
> rammer fx 5 % fejl. Det er den **iboende rang** — den "gulvlinje", læringen i
> bedste fald kan nå. Plot den lærte værdifejl oven på gulvlinjen: ligger den lærte
> fejl *over* gulvet, er flaskehalsen *læringen*, ikke rangen.

---

## 7. Konkrete eksperimenter og forventede resultater

- **Eksp. A — rang-sweep på gitteret.** CP med for lav rang fejler; så snart
  `R ≥` den iboende rang, matcher CP (slår ofte let) tabellen ved langt færre
  parametre. *Forventet overskrift:* "CP rang 4 matcher tabellen ved ~3× færre
  parametre."
- **Eksp. B — værdifejl-kurver.** CP konvergerer mod `Q*`; sammenlign plateauet på
  tværs af rang.
- **Eksp. C (valgfrit) — det svære miljø.** CP kan ikke komprimere et højrangs `Q*`
  → ingen parameterbesparelse → konklusion: lavrangs-struktur er en egenskab ved
  *miljøet*, ikke en gratis frokost.
- **Eksp. D (valgfrit) — ablation, der viser, at NLMS er nødvendigt.** Kør naiv SGD
  mod det normaliserede skridt; den naive divergerer. Validerer designet og giver en
  god figur.

---

## 8. Milepæle (selve køreplanen)

| | Milepæl | Leverance |
|---|---|---|
| **M1** | Tabulær Q-læring + gittermiljø + `Q*` via værdiiteration | baseline-læringskurve; et fungerende `Q*` |
| **M2** | CP-repræsentation: `entry`, `entry_grad` | **numerisk gradienttjek** består (se faldgruber) |
| **M3** | CP Q-lærings-opdatering med NLMS | CP-agenten lærer gitteret |
| **M4** | Rang-sweep + de fire sammenligningsfigurer | ydelse-mod-parametre-figuren |
| **M5** | Rapport + én udvidelse (svært miljø, eller SARSA-CP) | rapport |

Realistisk rækkefølge: M1 og M2 kan køre parallelt; M3 er det egentlige
forskningsskridt; M4 producerer resultaterne; M5 fortolker dem.

---

## 9. Hvad rapporten skal vise

En vellykket rapport indeholder: (1) ydelse-mod-parametre-figuren med tabellen som
referencepunkt, (2) værdifejl-kurver mod `Q*`, (3) en klar konklusion om *hvornår*
CP betaler sig (lavrangs miljø) og *hvornår ikke* (højrangs miljø), og (4) en
diskussion af, om den resterende fejl skyldes rang eller læring. Det er konkrete,
falsificerbare udsagn — præcis det, et godt empirisk projekt leverer.
