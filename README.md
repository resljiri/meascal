# Calibration Reliability System v6.3

Streamlit prototyp pro intervalově cenzorovanou Weibullovu analýzu kalibračních dat. OBLAST není součástí modelu.

## Co je nové ve v6.3

Hlavní změnou je **perzistentní registr modelů**. Přeučení už nevytváří pouze model v `st.session_state`. Každý nový fit se uloží jako samostatná verze:

```text
models/
├── active_model.json
├── caliper_v1/                 # původní importovaný model
└── registry/
    └── caliper_YYYYMMDD_HHMMSS/
        ├── model.joblib
        ├── metadata.json
        ├── CAT_eta.csv
        ├── CAT_shape.csv
        ├── combo_effects.csv
        ├── PROD_effects.csv
        └── TYP_effects.csv
```

`models/active_model.json` je jediný ukazatel, který říká, která verze se má při startu aplikace načíst. Aplikace tedy model při každém spuštění **nepřeučuje**.

Po refitu se:

1. vytvoří nová verzovaná složka,
2. uloží `model.joblib`, metadata a čitelné CSV koeficienty,
3. aktualizuje `models/active_model.json`,
4. nový model se ihned aktivuje,
5. vyčistí se cache starých predikcí.

V modulu **Přeučení modelu** lze také přepínat mezi dříve uloženými verzemi.

## Důležité pro Streamlit Community Cloud

Lokální disk běžícího Streamlit Cloud kontejneru není garantované trvalé úložiště. Model vytvořený přímo v cloudové aplikaci proto funguje po běžných rerunech, ale po rebootu nebo redeployi může zmizet, pokud nebyl zapsán do GitHub repozitáře.

Po novém refitu aplikace proto nabídne tlačítko **Stáhnout balíček aktivního modelu pro GitHub**. ZIP obsahuje jen:

```text
models/active_model.json
models/registry/<nová_verze>/...
```

Rozbal jej do kořene lokálního projektu a pak ve VS Code spusť:

```powershell
git add models
git commit -m "Persist active reliability model"
git push
```

Po redeployi Streamlit automaticky načte stejný model. Ten zůstane aktivní, dokud `active_model.json` nezměníš dalším refitem nebo ruční aktivací jiné verze.

### Nejjednodušší dlouhodobý workflow

Ještě čistší je přeučovat model lokálně ve VS Code, kde se artefakt rovnou uloží do `models/`, a pak provést běžný `git add/commit/push`. Streamlit Cloud pak slouží pouze jako predikční aplikace.

## Model

Adaptivní hierarchický location–shape Weibull:

```text
log(eta) = CAT + CLMARK×RNG + PROD + TYP(PROD)
log(k)   = CAT + CLMARK×RNG + PROD + TYP(PROD)
```

Efekty jsou aktivovány podle datové podpory a shrinkovány penalizací. Při novém refitu mohou částečné záznamy přispět k těm částem modelu, pro které mají známé kovariáty; například záznam bez TYP může stále informovat intercept, CAT, CLMARK×RNG a PROD.

Cílový interval je obecně

```text
t_R = eta * [-ln(R)]^(1/k)
```

a aplikace zobrazuje časy ve dnech i v letech.

## Spuštění

```powershell
pip install -r requirements.txt
streamlit run app.py
```

## Git struktura

Složky musí na GitHubu zůstat zachované:

```text
app.py
src/
config/
models/
assets/
.streamlit/
data/
tests/
```

Nenahrávej obsah `src/` přímo do kořene repozitáře.
