from __future__ import annotations
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def survival_diagnostic_figure(
    times,
    primary_survival,
    target_reliability,
    target_time,
    target_ci=None,
    primary_label="Hlavní výsledek",
    observed_df=None,
    show_raw=False,
    show_turnbull=False,
    turnbull_survival=None,
    comparison_survival=None,
    comparison_label=None,
    primary_ci_lower=None,
    primary_ci_upper=None,
    sentinel=99999.0,
    max_raw_records=750,
):
    """Generic survival plot. Primary curve may be hierarchical or a direct Weibull fit."""
    times=np.asarray(times,float)
    use_lower=bool(show_raw and observed_df is not None and len(observed_df))
    if use_lower:
        fig=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[0.78,0.22],vertical_spacing=0.04,
                          subplot_titles=("Survival křivka","Skutečná intervalově cenzorovaná data"))
        main_row,raw_row=1,2
    else:
        fig=make_subplots(rows=1,cols=1); main_row,raw_row=1,None

    if primary_ci_lower is not None and primary_ci_upper is not None:
        lo=np.asarray(primary_ci_lower,float); hi=np.asarray(primary_ci_upper,float)
        if len(lo)==len(times) and len(hi)==len(times):
            fig.add_trace(go.Scatter(x=times,y=hi,mode="lines",line=dict(width=0),hoverinfo="skip",showlegend=False,name="95% CI horní"),row=main_row,col=1)
            fig.add_trace(go.Scatter(x=times,y=lo,mode="lines",line=dict(width=0),fill="tonexty",fillcolor="rgba(99,110,250,0.15)",hoverinfo="skip",name="95% CI křivky"),row=main_row,col=1)
    fig.add_trace(go.Scatter(x=times,y=primary_survival,mode="lines",name=primary_label,line=dict(width=3)),row=main_row,col=1)
    if comparison_survival is not None:
        fig.add_trace(go.Scatter(x=times,y=comparison_survival,mode="lines",name=comparison_label or "Porovnání",line=dict(width=2,dash="dash")),row=main_row,col=1)
    if show_turnbull and turnbull_survival is not None:
        fig.add_trace(go.Scatter(x=times,y=turnbull_survival,mode="lines",name="Turnbull – data",line=dict(shape="hv",width=2)),row=main_row,col=1)

    rel_pct=100*float(target_reliability)
    rel_label=(f"{rel_pct:.0f}" if abs(rel_pct-round(rel_pct))<1e-8 else f"{rel_pct:.1f}")
    fig.add_hline(y=target_reliability,line_dash="dot",annotation_text=f"R = {rel_label} %",row=main_row,col=1)
    years=float(target_time)/365.25
    years_txt=f"{years:.1f}".replace(".",",")
    fig.add_vline(x=target_time,line_dash="dot",annotation_text=f"t{rel_label} = {target_time:.0f} d ({years_txt} roku)",row=main_row,col=1)
    if target_ci:
        fig.add_vrect(x0=target_ci[0],x1=target_ci[1],opacity=.10,line_width=0,annotation_text=f"95% CI t{rel_label}",row=main_row,col=1)

    if use_lower:
        obs=observed_df.copy()
        if len(obs)>max_raw_records:
            obs=obs.sample(max_raw_records,random_state=42).sort_values("_row_id")
        nok=obs[~obs["RIGHT_CENSORED"]].copy(); cens=obs[obs["RIGHT_CENSORED"]].copy()
        if len(nok):
            xs,ys,custom=[],[],[]; yvals=1.0+((np.arange(len(nok))%9)-4)*.025
            for y,(_,r) in zip(yvals,nok.iterrows()):
                xs += [float(r.Left),float(r.Right),None]; ys += [float(y),float(y),None]
                custom += [[str(r.get("_row_id","")),float(r.Left),float(r.Right),"NOK"],
                           [str(r.get("_row_id","")),float(r.Left),float(r.Right),"NOK"],[None,None,None,None]]
            fig.add_trace(go.Scatter(x=xs,y=ys,mode="lines",name=f"NOK intervaly ({len(nok)})",customdata=custom,
                                     hovertemplate="ID: %{customdata[0]}<br>L=%{customdata[1]:.0f} d<br>R=%{customdata[2]:.0f} d<br>%{customdata[3]}<extra></extra>",line=dict(width=2)),row=raw_row,col=1)
        if len(cens):
            yvals=.20+((np.arange(len(cens))%9)-4)*.025
            custom=np.column_stack([cens.get("_row_id","").astype(str),cens["Left"].to_numpy(float),np.repeat("T > L (pravostranně cenzorováno)",len(cens))])
            fig.add_trace(go.Scatter(x=cens["Left"],y=yvals,mode="markers",name=f"Cenzorováno ({len(cens)})",marker=dict(symbol="triangle-right",size=8),customdata=custom,
                                     hovertemplate="ID: %{customdata[0]}<br>L=%{customdata[1]:.0f} d<br>%{customdata[2]}<extra></extra>"),row=raw_row,col=1)
        fig.update_yaxes(tickmode="array",tickvals=[.2,1.0],ticktext=["CENS","NOK"],range=[-.05,1.25],title_text="Pozorování",row=raw_row,col=1)
        fig.update_xaxes(title_text="Čas [dny]",row=raw_row,col=1)
    else:
        fig.update_xaxes(title_text="Čas [dny]",row=main_row,col=1)
    fig.update_yaxes(title_text="R(t)",range=[0,1.01],row=main_row,col=1)
    fig.update_layout(height=720 if use_lower else 560,hovermode="closest",legend=dict(orientation="h"),margin=dict(t=70,b=40))
    return fig
