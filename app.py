from pathlib import Path
import time
import hashlib
import numpy as np
import pandas as pd
import yaml
import plotly.graph_objects as go
import streamlit as st

from src.data import load_calibration_data, apply_filters, model_coverage_summary
from src.model import AdaptiveWeibullModel
from src.model_registry import (
    active_descriptor, registry_token, load_active_model, save_versioned_model,
    list_versions, activate_version, activate_imported, make_persistence_bundle
)
from src.diagnostics import support_summary, profile_data_subset, current_profile_subset, fit_direct_weibull, turnbull_curve, direct_survival_ci
try:
    from src.diagnostics import batch_predict_fast
except ImportError:
    # Backward compatibility if Streamlit/GitHub temporarily contains an older diagnostics.py
    from src.diagnostics import batch_predict as batch_predict_fast
from src.plots import survival_diagnostic_figure

ROOT=Path(__file__).resolve().parent
CFG=yaml.safe_load((ROOT/'config/app.yaml').read_text(encoding='utf-8'))
FAMILY=CFG['families']['CALIPER']; ALL=AdaptiveWeibullModel.ALL_TOKEN; ALL_LABEL='Všechny / bez specifického efektu'
SENTINEL=FAMILY['model'].get('right_censor_sentinel',99999)

st.set_page_config(page_title='Calibration Reliability System',layout='wide')
css_path=ROOT/'assets/style.css'
if css_path.exists(): st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>",unsafe_allow_html=True)
st.title('Calibration Reliability System')

@st.cache_data
def get_data(): return load_calibration_data(ROOT/FAMILY['data_file'],SENTINEL)
@st.cache_resource
def get_persisted_model(token):
    # token is part of the cache key and changes whenever models/active_model.json changes
    return load_active_model(ROOT, FAMILY['model'], None)

def rel_text(rel):
    x=100*rel
    return f"{x:.0f}" if abs(x-round(x))<1e-9 else f"{x:.1f}"
def as_str_rng(v):
    try:
        x=float(v); return str(int(x)) if x.is_integer() else str(x)
    except Exception: return str(v)
def normalize_value(field,v): return ALL if v==ALL else (as_str_rng(v) if field=='RNG' else str(v))
def matching_subset(data,selections):
    out=data.copy()
    for field,val in selections.items():
        if val==ALL: continue
        if field=='TYP': out=out[out['TYP_STR'].astype(str)==str(val)]
        elif field=='RNG': out=out[pd.to_numeric(out['RNG'],errors='coerce')==float(val)]
        else: out=out[out[field].astype(str)==str(val)]
    return out
def domain_values(data,field):
    col='TYP_STR' if field=='TYP' else field; vals=data[col].dropna().unique().tolist()
    if field=='RNG': return sorted(vals,key=lambda x:float(x))
    return sorted(map(str,vals))
def selector(label,field,col,data,prior,valid_only,key,disabled=False):
    if disabled: return col.selectbox(label,[ALL],format_func=lambda x:ALL_LABEL,key=key,disabled=True)
    valid_df=matching_subset(data,prior); valid=set(normalize_value(field,x) for x in domain_values(valid_df,field))
    full=[normalize_value(field,x) for x in domain_values(data,field)]
    opts=[ALL]+(sorted(valid,key=str) if valid_only else sorted(set(full),key=str))
    if st.session_state.get(key) not in opts: st.session_state[key]=ALL
    def fmt(x):
        if x==ALL: return ALL_LABEL
        return f"⚠ {x} – kombinace zatím nepozorována" if (not valid_only and x not in valid) else str(x)
    return col.selectbox(label,opts,format_func=fmt,key=key)
def ci_text(ci,fmt='.0f'):
    return 'není dostupný' if not ci else f"{format(ci[0],fmt)} až {format(ci[1],fmt)}"

def days_years(v, decimals_days=0):
    if v is None or not np.isfinite(v): return '—'
    d=f"{float(v):,.{decimals_days}f}".replace(',', ' ')
    y=f"{float(v)/365.25:.1f}".replace('.', ',')
    return f"{d} d ({y} roku)"

def ci_days_years(ci):
    if not ci: return 'není dostupný'
    return f"{days_years(ci[0])} až {days_years(ci[1])}"

def direct_survival(direct,times): return np.exp(-(np.asarray(times,float)/direct['eta'])**direct['k'])

def confidence_adaptive(r, current_n=0, current_nok=0, exact_exists=True):
    score=0.0; reasons=[]
    if current_n >= 200: score += 2; reasons.append('n≥200')
    elif current_n >= 70: score += 1; reasons.append('n≥70')
    elif current_n < 30: score -= 1; reasons.append('n<30')
    if current_nok >= 100: score += 3; reasons.append('NOK≥100')
    elif current_nok >= 30: score += 2; reasons.append('NOK≥30')
    elif current_nok >= 10: score += 1; reasons.append('NOK≥10')
    else: score -= 1; reasons.append('NOK<10')
    levels={r.backoff.get('ETA'),r.backoff.get('K')}
    if 'CAT/POPULATION' in levels: score -= 2; reasons.append('populační back-off')
    elif 'COMBO' in levels: score -= .5; reasons.append('back-off na kombinaci')
    if not exact_exists: score -= 1; reasons.append('nepozorovaná přesná kombinace')
    if r.t_target_ci95 and r.t_target>0:
        ciw=(r.t_target_ci95[1]-r.t_target_ci95[0])/r.t_target
        if ciw <= .70: score += 1; reasons.append('úzké CI')
        elif ciw > 1.40: score -= 2; reasons.append('široké CI')
    if score >= 4: code,label='HIGH','Vysoká'
    elif score >= 2: code,label='MEDIUM','Střední'
    else: code,label='LOW','Nízká'
    return code,label,score,', '.join(reasons)

def confidence_direct(d):
    if d is None: return 'LOW','Nízká',0.0,'fit není dostupný'
    n,nok=d['n'],d['NOK']; ci=d.get('t_target_ci95'); width=(ci[1]-ci[0])/d['t_target'] if ci and d['t_target']>0 else None
    score=0.0; reasons=[]
    if n>=200: score+=2; reasons.append('n≥200')
    elif n>=70: score+=1; reasons.append('n≥70')
    elif n<30: score-=1; reasons.append('n<30')
    if nok>=100: score+=3; reasons.append('NOK≥100')
    elif nok>=30: score+=2; reasons.append('NOK≥30')
    elif nok>=10: score+=1; reasons.append('NOK≥10')
    else: score-=1; reasons.append('NOK<10')
    if width is not None:
        if width<=.70: score+=1; reasons.append('úzké CI')
        elif width>1.40: score-=2; reasons.append('široké CI')
    if score>=4: return 'HIGH','Vysoká',score,', '.join(reasons)
    if score>=2: return 'MEDIUM','Střední',score,', '.join(reasons)
    return 'LOW','Nízká',score,', '.join(reasons)
def badge(code,label):
    colors={'HIGH':'#15803d','MEDIUM':'#b45309','LOW':'#b91c1c'}
    st.markdown(f"<span style='background:{colors[code]};color:white;padding:0.28rem 0.65rem;border-radius:999px;font-weight:700'>Datová důvěra: {label}</span>",unsafe_allow_html=True)

def profile_key(profile):
    return tuple((k,str(profile.get(k))) for k in ["PROD","TYP","CAT","CLMARK","RNG"])

def row_ids_key(frame):
    if len(frame)==0: return ()
    return tuple(map(int,frame["_row_id"].tolist()))

def model_cache_token(model_name, model):
    # Name plus parameter vector checksum keeps cached results separated after refit/reload.
    th=getattr(model,"theta",None)
    if th is None: return str(model_name)
    a=np.asarray(th,float)
    return str(model_name)+":"+hashlib.sha1(a.tobytes()).hexdigest()[:12]

def cached_session_get(namespace,key):
    cache=st.session_state.setdefault(namespace,{})
    return cache.get(key)

def cached_session_put(namespace,key,value,max_items=80):
    cache=st.session_state.setdefault(namespace,{})
    cache[key]=value
    if len(cache)>max_items:
        for old in list(cache)[:len(cache)-max_items]: cache.pop(old,None)
    return value


df=get_data()
_active_token=registry_token(ROOT)
base_model, active_desc=get_persisted_model(_active_token)
base_model.set_population_context(df)
# Session model is used immediately after a refit. On a fresh app process the repository
# descriptor above determines the persistent active model.
model=st.session_state.get('active_model',base_model)
model_name=st.session_state.get('active_model_name', f"{active_desc.get('version','model')} ({'uložený artefakt' if active_desc.get('kind')=='joblib' else 'importovaný základ'})")

with st.sidebar:
    st.caption(f"Aktivní model: **{model_name}**")
    st.caption('95% CI: '+('✓ dostupný' if model.covariance is not None else '– chybí kovarianční matice'))
    target_pct=st.number_input('Cílová spolehlivost R [%]',min_value=50.0,max_value=99.9,value=float(st.session_state.get('target_pct',90.0)),step=.5,key='target_pct',help='Např. 90 % znamená výpočet t90; 95 % znamená t95.')
    target_rel=target_pct/100.0
    st.caption(f"Aktivní cíl: **t{rel_text(target_rel)}**")
    perf_mode=st.radio('Režim výpočtu',['Rychlý','Přesný'],horizontal=True,help='Bodový odhad je stejný. Liší se počet Monte Carlo vzorků pro CI a diagnostiku.')
    ci_mode=st.selectbox('CI v individuální analýze',['Rychlé CI','Přesné CI','Bez CI'],index=0)
    ci_samples=750 if ci_mode=='Rychlé CI' else (5000 if ci_mode=='Přesné CI' else 0)
    st.caption(f"Repozitářový aktivní artefakt: **{active_desc.get('version','—')}**")
    if active_desc.get('kind') != 'imported_csv' and st.button('Aktivovat původní caliper_v1'):
        activate_imported(ROOT, 'caliper_v1')
        st.session_state.pop('active_model',None); st.session_state.pop('active_model_name',None)
        st.session_state.pop('adaptive_prediction_cache',None)
        st.rerun()

page=st.sidebar.radio('Modul',['Dashboard','Výběr množiny','Jedno měřidlo','Hromadná analýza','Model a diagnostika','Přeučení modelu'])

if page=='Dashboard':
    c1,c2,c3,c4=st.columns(4); c1.metric('Záznamy',f"{len(df):,}".replace(',',' ')); c2.metric('NOK',f"{int(df.NOK.sum()):,}".replace(',',' ')); c3.metric('Výrobci',df.PROD.nunique()); c4.metric('Typy',df.PROD_TYP.nunique())
    st.info(f"Globálně nastavená cílová spolehlivost je {target_pct:.1f} %, tedy t{rel_text(target_rel)}. Nastavení v levém panelu používá individuální i hromadná analýza.")
    complete=int(df[['PROD','TYP','CAT','CLMARK','RNG']].notna().all(axis=1).sum())
    st.caption(f'Databáze obsahuje {len(df)} platných intervalových záznamů; kompletní profil všech pěti faktorů má {complete}. Při novém refitu mohou částečné záznamy informovat ty efekty, pro které mají známé kovariáty.')
    st.dataframe(df[['PROD','TYP_STR','CAT','CLMARK','RNG','Left','Right','NOK']].head(100),use_container_width=True)

elif page=='Výběr množiny':
    st.subheader('Dynamické filtry'); filters={}; cols=st.columns(3)
    for i,field in enumerate(FAMILY['filter_fields']+FAMILY.get('optional_filter_fields',[])):
        if field not in df.columns: continue
        vals=sorted(df[field].dropna().astype(str).unique().tolist()); filters[field]=cols[i%3].multiselect(field,vals)

    sub=apply_filters(df,filters)
    filtered_ids=set(sub['_row_id'].tolist())
    selected_set=set(st.session_state.get('manual_selection',[]))
    st.write(f"Po filtrech: **{len(sub)}** záznamů")

    b1,b2,b3,b4=st.columns([1,1,1,2])
    if b1.button('✓ Přidat vše z filtru',use_container_width=True,disabled=(len(sub)==0)):
        selected_set |= filtered_ids
        st.session_state['manual_selection']=sorted(selected_set)
        st.session_state['selection_editor_revision']=st.session_state.get('selection_editor_revision',0)+1
        st.rerun()
    if b2.button('− Odebrat vše z filtru',use_container_width=True,disabled=(len(sub)==0)):
        selected_set -= filtered_ids
        st.session_state['manual_selection']=sorted(selected_set)
        st.session_state['selection_editor_revision']=st.session_state.get('selection_editor_revision',0)+1
        st.rerun()
    if b3.button('× Zrušit celý výběr',use_container_width=True,disabled=(len(selected_set)==0)):
        st.session_state['manual_selection']=[]
        st.session_state['selection_editor_revision']=st.session_state.get('selection_editor_revision',0)+1
        st.rerun()

    selected_in_filter=len(selected_set & filtered_ids)
    b4.info(f"Vybráno ve filtru: **{selected_in_filter}/{len(sub)}**  |  Celkem vybráno: **{len(selected_set)}**")

    page_size=st.selectbox('Řádků na stránku',[100,250,500],index=0,key='selection_page_size')
    n_pages=max(1,int(np.ceil(len(sub)/page_size)))
    page_no=st.number_input('Stránka',min_value=1,max_value=n_pages,value=min(int(st.session_state.get('selection_page_no',1)),n_pages),step=1,key='selection_page_no')
    page_sub=sub.iloc[(int(page_no)-1)*page_size:int(page_no)*page_size].copy()
    page_ids=set(page_sub['_row_id'].tolist())
    st.caption(f'Zobrazeno {len(page_sub)} z {len(sub)} záznamů | stránka {int(page_no)}/{n_pages}')
    view=page_sub[['_row_id','PROD','TYP_STR','CAT','CLMARK','RNG','NOK']].copy()
    view.insert(0,'Vybrat',view['_row_id'].isin(selected_set))
    # Klíč editoru se mění při změně filtru nebo hromadném tlačítku, aby checkboxy vždy
    # přesně odrážely aktuální stav uložený v session_state.
    filter_signature='|'.join(f"{k}:{','.join(map(str,v))}" for k,v in sorted(filters.items()))
    revision=st.session_state.get('selection_editor_revision',0)
    editor_key=f"manual_selection_editor_{filter_signature}_{revision}"
    edited=st.data_editor(
        view,
        use_container_width=True,
        hide_index=True,
        key=editor_key,
        column_config={'Vybrat':st.column_config.CheckboxColumn(required=True)},
        disabled=['_row_id','PROD','TYP_STR','CAT','CLMARK','RNG','NOK']
    )

    # Ruční změny checkboxů se aplikují jen na aktuálně filtrovanou množinu.
    # Výběr záznamů mimo aktuální filtr zůstává zachován.
    edited_selected=set(edited.loc[edited.Vybrat,'_row_id'].tolist())
    selected_set=(selected_set-page_ids) | edited_selected
    st.session_state['manual_selection']=sorted(selected_set)
    st.caption('Checkboxy upravují pouze právě zobrazenou stránku; výběr z ostatních stránek a filtrů zůstává zachován.')
    st.success(f"Aktuálně vybráno celkem: {len(selected_set)} záznamů")

elif page=='Jedno měřidlo':
    st.subheader('Individuální analýza')
    valid_only=st.toggle('Zobrazovat pouze hodnoty, které tvoří existující kombinace',value=True)
    c1,c2,c3,c4,c5=st.columns(5)
    prod=selector('Výrobce','PROD',c1,df,{},valid_only,'single_prod')
    typ=selector('Typ','TYP',c2,df,{'PROD':prod},valid_only,'single_typ',disabled=(prod==ALL))
    prior={'PROD':prod,'TYP':typ}; cat=selector('CAT','CAT',c3,df,prior,valid_only,'single_cat'); prior['CAT']=cat
    clm=selector('CLMARK','CLMARK',c4,df,prior,valid_only,'single_clm'); prior['CLMARK']=clm
    rng=selector('RNG','RNG',c5,df,prior,valid_only,'single_rng')
    p={'PROD':prod,'TYP':typ,'CAT':cat,'CLMARK':clm,'RNG':rng}

    selected_obs=current_profile_subset(df,p); all_specific=all(v!=ALL for v in [prod,typ,cat,clm,rng])
    exact_exists=len(selected_obs)>0
    if all_specific and not exact_exists:
        st.error('Tato přesná kombinace faktorů v databázi neexistuje. Hierarchický model může dát extrapolační odhad, přímý fit této vrstvy však není možný.')
    elif not exact_exists:
        st.warning('Pro aktuálně zadanou množinu nejsou v databázi žádné záznamy.')
    else:
        st.caption(f"Aktuálním omezením odpovídá **{len(selected_obs)}** záznamů, z toho **{int(selected_obs.NOK.sum())} NOK**.")

    level_labels={'CURRENT':'Aktuálně zadaná množina','TYPE':'PROD + TYP','PROD':'Pouze výrobce PROD','COMBO':'CLMARK × RNG','ALL':'Všechny posuvky'}
    a1,a2=st.columns([1.2,1.4])
    analysis_mode=a1.radio('Zdroj hlavního výsledku',['ADAPTIVE','DIRECT'],format_func=lambda x:'Adaptivní hierarchický model' if x=='ADAPTIVE' else 'Přímý Weibull fit vybrané vrstvy',horizontal=False)
    data_level=a2.selectbox('Úroveň dat pro přímý fit / diagnostiku',list(level_labels),format_func=lambda x:level_labels[x],help='Při režimu Přímý Weibull fit tato vrstva určuje přímo hlavní výsledek, jeho CI i hlavní křivku.')
    observed=profile_data_subset(df,p,data_level)
    st.caption(f"Vybraná datová vrstva: **{len(observed)}** záznamů | NOK: **{int(observed.NOK.sum()) if len(observed) else 0}**")

    token=model_cache_token(model_name,model); pkey=profile_key(p)
    adapt_key=(token,pkey,round(target_rel,6),ci_samples)
    adaptive=cached_session_get('adaptive_prediction_cache',adapt_key)
    if adaptive is None:
        t0=time.perf_counter()
        adaptive=model.predict(p,reliability_ci=(ci_samples>0),n_samples=max(ci_samples,1),reliability=target_rel)
        cached_session_put('adaptive_prediction_cache',adapt_key,adaptive)
        st.session_state['last_adaptive_ms']=1000*(time.perf_counter()-t0)

    direct=None
    if analysis_mode=='DIRECT' and len(observed)>=2:
        dkey=(row_ids_key(observed),round(target_rel,6),ci_samples if ci_samples>0 else 0)
        direct=cached_session_get('direct_fit_cache',dkey)
        if direct is None:
            with st.spinner('Počítám přímý Weibull fit...'):
                t0=time.perf_counter()
                direct=fit_direct_weibull(observed,SENTINEL,reliability=target_rel,compute_ci=(ci_samples>0),n_samples=max(ci_samples,1))
                cached_session_put('direct_fit_cache',dkey,direct)
                st.session_state['last_direct_ms']=1000*(time.perf_counter()-t0)

    if analysis_mode=='DIRECT' and direct is None:
        st.error('Přímý Weibull fit vybrané vrstvy se nepodařilo odhadnout.')
        primary_eta=primary_k=primary_t=np.nan
    elif analysis_mode=='DIRECT':
        primary_eta,primary_k,primary_t=direct['eta'],direct['k'],direct['t_target']
    else:
        primary_eta,primary_k,primary_t=adaptive.eta,adaptive.k,adaptive.t_target

    m1,m2,m3=st.columns(3)
    m1.metric('η – scale',days_years(primary_eta))
    m2.metric('k – shape',('—' if not np.isfinite(primary_k) else f"{primary_k:.3f}"))
    m3.metric(f"t{rel_text(target_rel)}",days_years(primary_t))

    if analysis_mode=='DIRECT' and direct:
        c1,c2,c3=st.columns(3)
        c1.caption(f"95% CI η: **{ci_days_years(direct.get('eta_ci95'))}**")
        c2.caption(f"95% CI k: **{ci_text(direct.get('k_ci95'),'.3f')}**")
        c3.caption(f"95% CI t{rel_text(target_rel)}: **{ci_days_years(direct.get('t_target_ci95'))}**")
        conf=confidence_direct(direct); badge(conf[0],conf[1]); st.caption(f"Skóre datové důvěry: {conf[2]:.1f} — {conf[3]}.")
        st.info('Hlavní výsledek je samostatný intervalově cenzorovaný Weibull fit pouze vybrané datové vrstvy.')
    else:
        c1,c2,c3=st.columns(3)
        c1.caption(f"95% CI η: **{ci_days_years(adaptive.eta_ci95)}**")
        c2.caption(f"95% CI k: **{ci_text(adaptive.k_ci95,'.3f')}**")
        c3.caption(f"95% CI t{rel_text(target_rel)}: **{ci_days_years(adaptive.t_target_ci95)}**")
        conf=confidence_adaptive(adaptive,len(selected_obs),int(selected_obs.NOK.sum()) if len(selected_obs) else 0,exact_exists)
        badge(conf[0],conf[1]); st.caption(f"Skóre datové důvěry: {conf[2]:.1f} — {conf[3]}.")
        st.write('**Back-off úroveň:** η =',adaptive.backoff['ETA'],'; k =',adaptive.backoff['K'])
        if ci_samples==0:
            st.info('CI jsou v režimu Bez CI vypnuté kvůli rychlosti.')
        elif model.covariance is None:
            st.info('CI není u importovaného modelu dostupný. Přeučený model s kovarianční maticí CI poskytne.')

    st.subheader('Survival křivka a skutečná data')
    d1,d2,d3,d4=st.columns(4)
    show_ci_band=d1.checkbox('95% CI křivky',value=True,help='Bodový 95% interval spolehlivosti R(t) z nejistoty parametrů. Je dostupný jen při existující kovarianční matici.')
    show_raw=d2.checkbox('Skutečná vstupní data',value=False)
    show_turnbull=d3.checkbox('Turnbullova křivka',value=False)
    show_comparison=d4.checkbox('Druhý model',value=False,help='V adaptivním režimu dopočítá přímý Weibull fit z vybrané datové vrstvy.')

    if analysis_mode=='ADAPTIVE' and show_comparison and len(observed)>=2:
        dkey=(row_ids_key(observed),round(target_rel,6),ci_samples if ci_samples>0 else 0)
        direct=cached_session_get('direct_fit_cache',dkey)
        if direct is None:
            with st.spinner('Počítám přímý Weibull fit pro porovnání...'):
                t0=time.perf_counter()
                direct=fit_direct_weibull(observed,SENTINEL,reliability=target_rel,compute_ci=(ci_samples>0),n_samples=max(ci_samples,1))
                cached_session_put('direct_fit_cache',dkey,direct)
                st.session_state['last_direct_ms']=1000*(time.perf_counter()-t0)

    observed_max=0.0
    if len(observed):
        fr=observed.loc[observed.Right<SENTINEL,'Right']
        observed_max=max(float(observed.Left.max()),float(fr.max()) if len(fr) else 0.0)
    ref_t=primary_t if np.isfinite(primary_t) else adaptive.t_target
    max_t=max(2500.,adaptive.t85*1.4,ref_t*1.5,observed_max*1.05)
    times=np.linspace(0,max_t,550)
    adaptive_surv=model.survival(p,times)
    direct_surv=direct_survival(direct,times) if direct else None

    if analysis_mode=='DIRECT' and direct:
        primary_surv=direct_surv; primary_label=f"Přímý Weibull – {level_labels[data_level]}"; target_time=direct['t_target']; target_ci=direct.get('t_target_ci95')
        comparison_surv=adaptive_surv if show_comparison else None; comparison_label='Adaptivní hierarchický model'
    else:
        primary_surv=adaptive_surv; primary_label='Adaptivní hierarchický model'; target_time=adaptive.t_target; target_ci=adaptive.t_target_ci95
        comparison_surv=direct_surv if show_comparison and direct is not None else None; comparison_label=f"Přímý Weibull – {level_labels[data_level]}"

    curve_ci=None
    if show_ci_band and ci_samples>0:
        curve_key=('DIRECT' if analysis_mode=='DIRECT' else 'ADAPTIVE',token,pkey,row_ids_key(observed) if analysis_mode=='DIRECT' else (),round(target_rel,6),round(max_t,2),len(times),ci_samples)
        curve_ci=cached_session_get('curve_ci_cache',curve_key)
        if curve_ci is None:
            with st.spinner('Počítám 95% CI křivky...'):
                if analysis_mode=='DIRECT':
                    curve_ci=direct_survival_ci(direct,times,n_samples=max(ci_samples,100))
                else:
                    curve_ci=model.survival_ci(p,times,n_samples=max(ci_samples,100),reliability=target_rel)
                cached_session_put('curve_ci_cache',curve_key,curve_ci,max_items=30)
    if show_ci_band and curve_ci is None:
        st.caption('95% CI křivky nelze zobrazit: pro aktivní model/fit není dostupná kovarianční matice nebo jsou CI vypnuté.')

    tb=None
    if show_turnbull and len(observed):
        tb_key=(row_ids_key(observed),round(float(max_t),3),len(times)); tb=cached_session_get('turnbull_cache',tb_key)
        if tb is None:
            with st.spinner('Počítám Turnbullův odhad...'):
                t0=time.perf_counter(); tb=turnbull_curve(observed,times,SENTINEL)
                cached_session_put('turnbull_cache',tb_key,tb,max_items=30)
                st.session_state['last_turnbull_ms']=1000*(time.perf_counter()-t0)

    fig=survival_diagnostic_figure(times,primary_surv,target_rel,target_time,target_ci,primary_label,
        observed_df=observed,show_raw=show_raw,show_turnbull=show_turnbull,turnbull_survival=tb,
        comparison_survival=comparison_surv,comparison_label=comparison_label,
        primary_ci_lower=(curve_ci[0] if curve_ci is not None else None),primary_ci_upper=(curve_ci[1] if curve_ci is not None else None),
        sentinel=SENTINEL,max_raw_records=FAMILY.get('visualization',{}).get('max_raw_records',750))
    st.plotly_chart(fig,use_container_width=True)
    if show_raw:
        st.caption('NOK jsou intervaly [L,R]; pravostranně cenzorované záznamy jsou značeny ▶ v čase L.')

    tabs=st.tabs(['Datová podpora','Použité efekty / rozklad','Vysvětlení'])
    with tabs[0]:
        st.dataframe(pd.DataFrame(adaptive.support).T,use_container_width=True)
        for w in adaptive.warnings: st.warning(w)
    with tabs[1]:
        ce=pd.DataFrame({'Efekt':list((adaptive.contributions_eta or {}).keys()),'Příspěvek log(η)':list((adaptive.contributions_eta or {}).values())})
        ck=pd.DataFrame({'Efekt':list((adaptive.contributions_k or {}).keys()),'Příspěvek log(k)':list((adaptive.contributions_k or {}).values())})
        q1,q2=st.columns(2); q1.dataframe(ce,use_container_width=True,hide_index=True); q2.dataframe(ck,use_container_width=True,hide_index=True)
        q1.caption(f"Součet LPη = {sum((adaptive.contributions_eta or {}).values()):.4f} → η = {days_years(adaptive.eta)}")
        q2.caption(f"Součet LPk = {sum((adaptive.contributions_k or {}).values()):.4f} → k = {adaptive.k:.3f}")
    with tabs[2]:
        st.markdown(f"""- **η** posouvá Weibullovu křivku v čase; při t=η je R≈36,8 %.
- **k** určuje tvar hazardu: k<1 klesající, k≈1 konstantní, k>1 rostoucí.
- **t{rel_text(target_rel)}** je čas, kdy R(t)={target_pct:.1f} %.
- **95% CI křivky** ukazuje bodovou nejistotu odhadované R(t) způsobenou nejistotou parametrů.
- **Datová důvěra** nyní kombinuje n, počet NOK, back-off, existenci přesné kombinace a šířku CI, pokud je dostupná.""")

    st.caption(f"Výkon: bodová/CI predikce {st.session_state.get('last_adaptive_ms',0):.0f} ms | poslední přímý fit {st.session_state.get('last_direct_ms',0):.0f} ms | poslední Turnbull {st.session_state.get('last_turnbull_ms',0):.0f} ms.")

elif page=='Hromadná analýza':
    st.subheader(f"Hromadná analýza – cíl t{rel_text(target_rel)}")
    use_manual=st.checkbox('Použít ruční výběr z modulu Výběr množiny',value=True)
    sub=df[df._row_id.isin(st.session_state['manual_selection'])].copy() if (use_manual and st.session_state.get('manual_selection')) else df.copy()
    st.write(f"Množina: **{len(sub)}** záznamů")
    batch_ci=st.selectbox('Nejistota v hromadné analýze',['Bez CI – nejrychlejší'],help='v6 zatím záměrně počítá batch bodové odhady bez Monte Carlo CI; přesné CI zůstává v individuální analýze.')
    if st.button('Spustit hromadnou analýzu',type='primary'):
        with st.spinner('Počítám unikátní profily...'):
            t0=time.perf_counter()
            pred=batch_predict_fast(model,sub,reliability=target_rel)
            out=sub.merge(pred,on='_row_id'); st.session_state['batch_output']=out; st.session_state['batch_rel']=target_rel
            st.session_state['batch_ms']=1000*(time.perf_counter()-t0)
    if 'batch_output' in st.session_state:
        out=st.session_state['batch_output']; used_rel=st.session_state.get('batch_rel',target_rel)
        c1,c2,c3=st.columns(3); c1.metric(f"Medián t{rel_text(used_rel)}",f"{out.t_target.median():.0f} d"); c2.metric('Medián k',f"{out.k.median():.3f}"); c3.metric('N záznamů',len(out))
        st.caption(f"Poslední hromadný výpočet: {st.session_state.get('batch_ms',0):.0f} ms. Každý unikátní PROD/TYP/CAT/CLMARK/RNG profil se počítá jen jednou.")
        fig=go.Figure(go.Histogram(x=out.t_target,nbinsx=40)); fig.update_layout(title=f"Distribuce t{rel_text(used_rel)}",xaxis_title='Dny',yaxis_title='Počet'); st.plotly_chart(fig,use_container_width=True)
        st.dataframe(support_summary(out.rename(columns={'eta_level':'eta_level','k_level':'k_level'})),use_container_width=True)
        st.dataframe(out[['_row_id','PROD','TYP_STR','CAT','RNG','eta','k','t_target','eta_level','k_level','warning']],use_container_width=True)
        st.download_button('Export CSV',out.to_csv(index=False).encode('utf-8'),'batch_analysis.csv','text/csv')

elif page=='Model a diagnostika':
    st.subheader('Aktuální model'); st.write('Location–shape Weibull s adaptivním back-offem.'); st.write(f"Aktivní model: **{model_name}**"); st.write('Kovarianční matice / CI:','dostupná' if model.covariance is not None else 'nedostupná')
    st.markdown('**Využitelná datová podpora pro nový refit**')
    st.dataframe(model_coverage_summary(df),use_container_width=True,hide_index=True)
    st.caption('Importovaný model byl původně fitován pouze na kompletních profilech. Refit v této verzi zachovává i částečné záznamy: např. záznam bez TYP může stále přispět k interceptu, CAT, CLMARK×RNG a PROD, pokud jsou tyto hodnoty známé.')
    st.dataframe(model.combo,use_container_width=True)
    c1,c2=st.columns(2)
    with c1:
        fig=go.Figure(go.Histogram(x=model.prod.u_eta,nbinsx=20)); fig.update_layout(title='PROD efekty pro η'); st.plotly_chart(fig,use_container_width=True)
    with c2:
        fig=go.Figure(go.Histogram(x=model.prod.u_k,nbinsx=20)); fig.update_layout(title='PROD efekty pro k'); st.plotly_chart(fig,use_container_width=True)

elif page=='Přeučení modelu':
    st.subheader('Přeučení a registr modelů')
    st.info('Model se po fitu ukládá jako verzovaný artefakt a stává se aktivním. Běžné reruny aplikace ho znovu netrénují. Na Streamlit Community Cloud je však disk kontejneru dočasný; aby nový model přežil reboot/redeploy, je potřeba uložené soubory z models/ commitnout do GitHubu.')
    st.dataframe(model_coverage_summary(df),use_container_width=True,hide_index=True)

    current=active_descriptor(ROOT)
    st.markdown(f"**Aktivní uložený model:** `{current.get('version','—')}`  ")
    st.caption(f"Zdroj: {current.get('source','—')} | artefakt: {current.get('path','—')}")

    versions=list_versions(ROOT)
    if versions:
        labels=[v.get('version','?') for v in versions]
        chosen=st.selectbox('Dříve uložené verze v registru',labels,index=0)
        cva,cvb=st.columns([1,2])
        if cva.button('Aktivovat vybranou verzi'):
            activate_version(ROOT,chosen)
            st.session_state.pop('active_model',None); st.session_state.pop('active_model_name',None)
            for k0 in ['adaptive_prediction_cache','direct_fit_cache','turnbull_cache']:
                st.session_state.pop(k0,None)
            st.rerun()
        with cvb:
            meta=next((v for v in versions if v.get('version')==chosen),{})
            fit=meta.get('fit') or {}
            st.caption(f"n={fit.get('n','—')}, NOK={fit.get('NOK','—')}, parametry={fit.get('n_params',meta.get('n_parameters','—'))}, CI={'ano' if meta.get('covariance_available') else 'ne'}")

    st.divider()
    st.markdown('### Nový refit')
    compute_cov=st.checkbox('Po fitu dopočítat kovarianční matici a CI',value=True)
    version_note=st.text_input('Poznámka k verzi',value='',placeholder='např. data 2026-08, nový refit')
    if st.button('Přeučit na aktuálních datech a uložit novou verzi',type='primary'):
        with st.spinner('Probíhá fit a případný výpočet kovarianční matice...'):
            candidate=AdaptiveWeibullModel(FAMILY['model']).fit(df,compute_covariance=compute_cov)
            # Lightweight fingerprint for auditability; no raw data are embedded in metadata.
            data_sig=hashlib.sha1(pd.util.hash_pandas_object(df[['Left','Right','PROD','TYP','CAT','CLMARK','RNG']],index=True).values.tobytes()).hexdigest()[:16]
            desc,folder=save_versioned_model(ROOT,candidate,source=(version_note.strip() or 'refit from application'),data_fingerprint=data_sig,activate=True)
            candidate.set_population_context(df)
            st.session_state['active_model']=candidate
            st.session_state['active_model_name']=f"{desc['version']} – nově přeučený"
            st.session_state['candidate_fit']=candidate.fit_result
            for k0 in ['adaptive_prediction_cache','direct_fit_cache','turnbull_cache']:
                st.session_state.pop(k0,None)
            bundle=make_persistence_bundle(ROOT,desc,ROOT/'models'/'active_model_bundle.zip')
            st.session_state['last_model_bundle']=str(bundle)
            st.session_state['last_model_desc']=desc
        st.success(f"Nová verze **{desc['version']}** byla uložena a aktivována. Do dalšího refitu ji aplikace používá bez nového učení.")
        if candidate.covariance is not None:
            st.success('Kovarianční matice byla vypočtena; CI jsou dostupná.')

    bundle_path=st.session_state.get('last_model_bundle')
    if bundle_path and Path(bundle_path).exists():
        data=Path(bundle_path).read_bytes()
        st.download_button('Stáhnout balíček aktivního modelu pro GitHub',data=data,file_name='active_model_bundle.zip',mime='application/zip')
        st.caption('Pro trvalé zachování na Streamlit Cloud rozbal tento ZIP do kořene lokálního projektu (obsahuje models/active_model.json a models/registry/<verze>/...), potom proveď git add models → commit → push. Po redeployi se tato verze automaticky načte.')

    if 'candidate_fit' in st.session_state:
        with st.expander('Výsledek posledního fitu'):
            st.json(st.session_state['candidate_fit'])

