# v6.1 hotfix

Tato verze přidává zpětně kompatibilní import `batch_predict_fast`. Pokud je na Streamlit Cloud dočasně starší `src/diagnostics.py`, aplikace nespadne při importu a použije `batch_predict`. Pro plný výkon musí být na GitHubu nahraná i aktuální v6.1 složka `src/`.

# Calibration Reliability System — v5

Prototyp pro posuvná měřítka, připravený pro další rodiny měřidel.

## Hlavní novinky ve v5

- globálně volitelná **cílová spolehlivost R** (např. 90 %, 95 %, 99 %); aplikace místo fixního t90 počítá obecné `tR`,
- v individuální analýze lze zvolit hlavní režim:
  - **adaptivní hierarchický model**,
  - **přímý intervalově cenzorovaný Weibull fit vybrané datové vrstvy**,
- při přímém režimu změna datové vrstvy mění současně hlavní `eta`, `k`, `tR`, 95% CI i hlavní křivku,
- přímý Weibull fit má vlastní přibližné 95% CI z Hessianu propagované Monte Carlo simulací,
- volitelné zobrazení druhého modelu pro porovnání,
- vizuální ukazatel **Datová důvěra: HIGH / MEDIUM / LOW**,
- tabulka datové podpory a samostatný rozklad příspěvků do `log(eta)` a `log(k)`,
- cílová spolehlivost se používá i v hromadné analýze a exportu.

## Individuální analýza

1. Vyberte faktory. Volba `Všechny / bez specifického efektu` použije pooled hierarchický příspěvek.
2. Zvolte zdroj hlavního výsledku.
3. Vyberte datovou vrstvu pro přímý fit/diagnostiku (`aktuální množina`, `PROD+TYP`, `PROD`, `CLMARK×RNG`, `všechny posuvky`).
4. Aplikace zobrazí hlavní `eta`, `k`, `tR`, jejich 95% CI, datovou důvěru, survival křivku a volitelně raw intervaly/Turnbull/porovnávací model.

### Přímý Weibull fit

Není to hierarchický model s vypnutými faktory. Jde o nový samostatný dvouparametrický Weibull fit pouze dat z vybrané vrstvy. Proto se po změně vrstvy může výrazně změnit výsledek i CI.

## CI

- Adaptivní model: CI jsou dostupná, pokud aktivní model obsahuje kovarianční matici (typicky po refitu s volbou výpočtu covariance).
- Přímý fit: CI se odhadují z numerického Hessianu parametrů `(log eta, log k)` a propagují Monte Carlo simulací do `tR`.

## Spuštění

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m streamlit run app.py
```

## Design

- `.streamlit/config.toml` — základní theme,
- `assets/style.css` — vlastní CSS,
- `src/plots.py` — Plotly grafy,
- `app.py` — layout/UI,
- `src/model.py` — statistické jádro.


## Novinka v5 – hromadný ruční výběr

V modulu **Výběr množiny** jsou nad tabulkou nově tlačítka:

- **Přidat vše z filtru** – přidá do ručního výběru všechny právě filtrované záznamy.
- **Odebrat vše z filtru** – odebere z ručního výběru všechny právě filtrované záznamy.
- **Zrušit celý výběr** – vymaže ruční výběr napříč celou databází.

Ruční checkboxy nadále fungují po jednotlivých záznamech. Výběr záznamů mimo aktuální filtr zůstává zachován, takže lze postupně skládat množinu z více různých filtrů. Panel ukazuje počet vybraných záznamů v aktuálním filtru i celkový počet vybraných záznamů.

## v6 – výkonová optimalizace

V6 nemění statistickou logiku modelu. Optimalizuje způsob, jakým Streamlit výpočty spouští a znovu používá:

- data a importovaný model používají `st.cache_data` / `st.cache_resource`,
- individuální adaptivní predikce a CI jsou uloženy v session cache podle modelu, profilu, cílové spolehlivosti a počtu MC vzorků,
- přímý Weibull fit se v adaptivním režimu nepočítá, dokud uživatel nezapne porovnávací model,
- Turnbull se počítá jen po zapnutí a stejný výsledek se znovu použije,
- `Rychlé CI` používá 750 Monte Carlo drawů, `Přesné CI` 5000, `Bez CI` žádné,
- hromadná analýza vyhodnotí každý unikátní PROD/TYP/CAT/CLMARK/RNG profil jen jednou,
- tabulka ručního výběru je stránkovaná (100/250/500 řádků),
- po přeučení se výpočetní cache invaliduje.

Výchozí doporučení pro Streamlit Community Cloud: `Rychlý` + `Rychlé CI`; Turnbull a přímý fit zapínat jen při diagnostice.

## Změny ve v6.2

- 95% bodový interval spolehlivosti lze vykreslit jako plochu kolem survival křivky, pokud má aktivní model/fit kovarianční matici.
- Časy η a tR se v individuální analýze zobrazují ve dnech i v letech (1 desetinné místo).
- Datová důvěra kombinuje velikost aktuální množiny, počet NOK, back-off, existenci přesné kombinace a šířku CI, pokud je dostupná.
- Ovládací prvky grafu jsou přímo nad grafem; datová podpora, rozklad efektů a vysvětlení jsou až pod grafem.
- Odstraněn textový popisek verze v záhlaví a spodní technická nápověda k souborům aplikace.
- Loader již nevyřazuje celý záznam jen proto, že chybí některá kovariáta. Při novém refitu může záznam bez TYP stále informovat např. intercept, CAT, CLMARK×RNG a PROD, jsou-li tyto hodnoty známé. Importovaný model zůstává původním modelem a jeho koeficienty se tím automaticky nemění.
