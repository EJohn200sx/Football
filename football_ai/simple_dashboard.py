"""Match-centred Streamlit dashboard; real feeds and explicit model assumptions."""
from __future__ import annotations
from datetime import datetime, timezone
from html import escape
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import poisson
import streamlit as st

from football_ai.config import Settings
from football_ai.database import connect, load_matches
from football_ai.live_data import FootballAPI, ProviderError
from football_ai.community_data import CommunityFootball, LEAGUES
from football_ai.team_insights import STAT_NAMES, find_teams, recent_matches, summarize, matchup_estimates
from football_ai.match_forecast import count_scenarios, score_scenarios
from football_ai.player_insights import PLAYER_EVENTS, rank_players
from football_ai.player_dashboard import player_spotlights, show_players
from football_ai.share_report import export_html

GREEN, BLUE, MUTED = "#b5f36d", "#73bcff", "#7e91a4"
SOURCES = ["Gratis · sin clave", "API-Football · con clave", "Histórico local"]
ZONES = {"UTC": "UTC", "Perú / Colombia": "America/Lima", "España peninsular": "Europe/Madrid",
         "México central": "America/Mexico_City", "Argentina": "America/Argentina/Buenos_Aires"}


@st.cache_resource(show_spinner=False)
def public_client():
    """Shared only for the public, credential-free source."""
    return CommunityFootball(min_interval=0.6)


def fmt(value, suffix=""):
    return "—" if value is None or pd.isna(value) else f"{value:.2f}{suffix}"


def html(text):
    st.markdown(text, unsafe_allow_html=True)


def chart(fig, height=330):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial, sans-serif", color="#bdcad7", size=12), height=height,
        margin=dict(l=12,r=12,t=25,b=20), legend=dict(orientation="h",y=1.13,x=0),
        colorway=[GREEN, BLUE], hoverlabel=dict(bgcolor="#18232e"))
    fig.update_xaxes(gridcolor="#26323d", zeroline=False)
    fig.update_yaxes(gridcolor="#26323d", zeroline=False)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def badge(name, url=None):
    # Locally rendered monograms: no broken remote logos or third-party requests.
    words = [w for w in name.split() if w.casefold() not in {"fc", "cf", "afc"}]
    initials = "".join(w[0] for w in words[:3]) if len(words)>1 else name[:3]
    return f'<div class="fi-monogram">{escape(initials.upper())}</div>'


def form_chips(form):
    return '<span class="fi-form">'+''.join(f'<b class="fi-{r}">{r}</b>' for r in form.split() if r in "VED")+'</span>'


def hero(fixture, summaries, zone):
    kickoff = pd.to_datetime(fixture["date"], utc=True).tz_convert(zone)
    home, away = escape(fixture["home_team"]), escape(fixture["away_team"])
    clock = kickoff.strftime("%H:%M") if fixture.get("time_confirmed", True) else "Por confirmar"
    html(f'''<div class="fi-hero"><div class="fi-eyebrow">Próximo registrado · {escape(fixture['competition'])}</div>
<div class="fi-fixture"><div class="fi-team">{badge(fixture['home_team'],fixture.get('home_logo'))}
<div class="fi-name">{home}</div><div class="fi-sub">LOCAL</div></div>
<div class="fi-vs">{kickoff:%d / %m / %Y}<div class="fi-time">{clock}</div>{escape(zone)}<br>VS</div>
<div class="fi-team away">{badge(fixture['away_team'],fixture.get('away_logo'))}
<div class="fi-name">{away}</div><div class="fi-sub">VISITANTE</div></div></div>
<div class="fi-bottom"><span>{home} &nbsp; {form_chips(summaries[0]['form'])}</span>
<span>{escape(fixture.get('venue') or 'Estadio no informado')}</span>
<span>{away} &nbsp; {form_chips(summaries[1]['form'])}</span></div></div>''')


def show_team(team, frame):
    summary = summarize(frame)
    st.subheader(team)
    st.caption(f"{summary['n']} partidos · forma del más antiguo al más reciente · V victoria / E empate / D derrota")
    if frame.empty:
        st.info("No hay partidos finalizados de este equipo en la ventana consultada.")
        return
    cols = st.columns(4)
    for col,key in zip(cols,["goals_for","goals_against","corners","yellow_cards"]):
        stat=summary["metrics"][key]
        col.metric(STAT_NAMES[key],fmt(stat["mean"]))
        col.caption(f"Promedio · {stat['available']}/{stat['total']} con dato")
    data = frame.assign(Fecha=lambda d: d.date.astype(str))
    fig = go.Figure()
    for key, color in [("goals_for", GREEN),("goals_against", BLUE)]:
        fig.add_bar(x=data.Fecha,y=data[key],name=STAT_NAMES[key],marker_color=color)
    chart(fig,260)
    st.dataframe(frame.rename(columns={**STAT_NAMES,"date":"Fecha","opponent":"Rival","venue":"Localía","result":"Resultado"}),
                 hide_index=True,width="stretch")


def show_forecasts(estimates, home, away, rankings=None):
    scores=score_scenarios(estimates["home_goals"],estimates["away_goals"])
    st.markdown("### Posibles resultados del partido")
    st.caption("PROYECCIÓN ORIENTATIVA · 90 MINUTOS · sin calibración predictiva")
    if scores:
        values=[scores["home_win"],scores["draw"],scores["away_win"]]
        html('<div class="fi-prob fi-prob-main">'+''.join(
            f'<div class="{side}"><span>{escape(name)}</span><strong>{v:.1%}</strong><small>{label}</small></div>'
            for name,v,side,label in zip([home,"Empate",away],values,["home","draw","away"],
                                        ["Victoria local","Igualdad","Victoria visitante"]))+'</div>')
        html('<div class="fi-result-bar">'+''.join(f'<div style="width:{v*100:.5f}%;background:{c}"></div>'
             for v,c in zip(values,[GREEN,MUTED,BLUE]))+'</div>')
        st.markdown("**Marcadores con mayor probabilidad en el modelo**")
        html('<div class="fi-score-grid">'+''.join(
            f'<div class="fi-score"><strong>{v["home"]}–{v["away"]}</strong><small>{v["probability"]:.1%}</small></div>'
            for v in scores['scores'])+'</div>')
        st.caption(f"Orden local–visitante. El resto de marcadores reúne {1-sum(v['probability'] for v in scores['scores']):.1%} de probabilidad.")
    scenarios=count_scenarios(estimates)
    html('<div class="fi-section">¿Cuántos podrían producirse?</div>')
    cards=[]
    for key,label in [("goals","Goles estimados"),("corners","Córners estimados"),("yellow_cards","Amarillas estimadas"),("shots","Tiros estimados")]:
        v=scenarios.get(key)
        if v:
            cards.append(f'<div class="fi-kpi"><div class="fi-kpi-label">{label} · total</div><div class="fi-kpi-value">{v["total"]:.1f}</div>'
                         f'<div class="fi-kpi-note">Rango: {v["low"]}–{v["high"]}</div><div class="fi-kpi-pair">Local {v["home"]:.1f} · Visitante {v["away"]:.1f}</div></div>')
        else:
            cards.append(f'<div class="fi-kpi"><div class="fi-kpi-label">{label}</div><div class="fi-kpi-value">—</div><div class="fi-kpi-note">Muestra insuficiente</div></div>')
    html('<div class="fi-kpi-grid">'+''.join(cards)+'</div>')
    st.caption('Los rangos contienen al menos el 80% de la masa del modelo Poisson, no una garantía de acierto.')
    if scores:
        a,b=st.columns(2)
        a.metric("Ambos equipos marcan",f"{scores['both_score']:.1%}")
        b.metric("3 o más goles en total",f"{scores['at_least_three']:.1%}")
    if rankings is not None:
        player_spotlights(rankings)
    if scenarios:
        with st.expander("Explorar distribuciones y más estadísticas"):
            choice=st.selectbox("Estadística del partido",list(scenarios),format_func=lambda k:scenarios[k]["label"])
            v=scenarios[choice]
            max_n=max(4,int(poisson.ppf(.995,v["total"])))
            x=list(range(max_n+1)); y=poisson.pmf(x,v["total"])*100
            fig=go.Figure(go.Bar(x=x,y=y,marker_color=[GREEN if v["low"]<=n<=v["high"] else "#344a3e" for n in x],
                hovertemplate="Cantidad: %{x}<br>Probabilidad: %{y:.1f}%<extra></extra>"))
            fig.update_layout(xaxis_title=f"{v['label']} de ambos equipos",yaxis_title="Probabilidad del modelo (%)")
            chart(fig,290)
            table=[{"Estadística":v["label"],home:round(v["home"],2),away:round(v["away"],2),
                    "Total estimado":round(v["total"],2),"Rango del modelo (≥80%)":f"{v['low']}–{v['high']}"} for v in scenarios.values()]
            st.dataframe(pd.DataFrame(table),hide_index=True,width="stretch")


def show_comparison(names, summaries):
    st.markdown("#### Promedios recientes, frente a frente")
    keys=["goals_for","goals_against","corners","yellow_cards","red_cards","shots","shots_on_target","fouls","possession","xg"]
    rows=[]
    for key in keys:
        a,b=[s["metrics"][key] for s in summaries]
        rows.append({"Estadística":STAT_NAMES[key],names[0]:a["mean"],names[1]:b["mean"],
                     "Cobertura local":f"{a['available']}/{a['total']}","Cobertura visitante":f"{b['available']}/{b['total']}"})
    selected=st.multiselect("Comparar en el gráfico",keys,default=["goals_for","corners","yellow_cards","shots_on_target"],format_func=lambda k:STAT_NAMES[k])
    fig=go.Figure()
    for name,summary,color in zip(names,summaries,[GREEN,BLUE]):
        fig.add_bar(name=name,x=[summary["metrics"][k]["mean"] for k in selected],y=[STAT_NAMES[k] for k in selected],
                    orientation="h",marker_color=color)
    fig.update_layout(barmode="group",xaxis_title="Promedio por partido (posesión en %)",yaxis=dict(autorange="reversed"))
    if selected:
        chart(fig,max(260,len(selected)*65))
    st.dataframe(pd.DataFrame(rows).style.format(precision=2,na_rep="—"),hide_index=True,width="stretch")
    st.caption("El signo — indica un dato no disponible. El xG sólo se muestra cuando existe en la fuente; "
               "no es lo mismo que los goles estimados para el próximo partido.")


def show_match_report(report, count, venue_only, now, zone="UTC"):
    for warning in report.get("warnings",[]):
        st.warning(warning)
    fixture=report.get("fixture")
    if not fixture:
        st.info("Prueba otra liga o vuelve a consultar cuando la fuente actualice su calendario.")
        return
    if pd.to_datetime(fixture["date"],utc=True)<=pd.Timestamp(now):
        st.info("El encuentro guardado ya comenzó. Pulsa Actualizar consulta para buscar el siguiente.")
        return
    home,away=fixture["home_team"],fixture["away_team"]
    competition=None if "coverage" in report else fixture["competition"]
    frames=[recent_matches(report["history"],team,competition,now,count,side if venue_only else None)
            for team,side in [(home,"home"),(away,"away")]]
    summaries=[summarize(f) for f in frames]
    stale=any(f.empty or (pd.Timestamp(now).tz_convert("UTC").normalize().tz_localize(None)-pd.Timestamp(f.date.max())).days>90 for f in frames)
    estimates=matchup_estimates(*summaries,stale=stale)
    hero(fixture,summaries,zone)
    st.caption(report.get("scope","Próximo encuentro programado devuelto por API-Football; sujeto a su cobertura."))
    if stale:
        st.warning("Falta historia de los últimos 90 días para algún equipo. No se calculan estimaciones con datos antiguos.")
    player_data=report.get("players",pd.DataFrame())
    rankings={key:rank_players(player_data,report["history"],fixture,now,key) for key in PLAYER_EVENTS}
    tabs=st.tabs(["Resumen","Jugadores","Equipos","Historial","Compartir","Método"])
    with tabs[0]:
        show_forecasts(estimates,home,away,rankings)
    with tabs[1]:
        show_players(report,rankings)
    with tabs[2]:
        show_comparison([home,away],summaries)
    with tabs[3]:
        for name,frame in zip([home,away],frames):
            show_team(name,frame)
    with tabs[4]:
        st.subheader("Comparte esta previa")
        st.write("Descarga un informe que cualquiera puede abrir en su navegador, sin instalar Python. Incluye el partido, los escenarios y los jugadores disponibles.")
        snapshot=export_html(report,estimates,summaries,rankings,zone,venue_only)
        st.download_button("Descargar informe para compartir",snapshot.encode("utf-8"),file_name=f"previa_{fixture['id']}.html",mime="text/html",type="primary")
        st.caption("Es una copia con fecha de consulta: no se actualiza. Puedes enviarla como archivo o abrirla y usar Imprimir → Guardar como PDF.")
        st.markdown("#### Compartir la web con un enlace")
        st.write("Para que otras personas busquen sus equipos necesitas publicar la aplicación. El proyecto incluye la carpeta publicar_web y la guía COMPARTIR.md, con los pasos de GitHub y Streamlit Community Cloud.")
        st.link_button("Abrir Streamlit Community Cloud","https://share.streamlit.io/")
        st.caption("La dirección 127.0.0.1 sólo funciona en tu equipo. Esta pantalla no publica la aplicación por sí misma.")
    with tabs[5]:
        st.markdown("#### Datos consultados")
        st.write(f"Fuente: **{report.get('source','API-Football')}**")
        st.write(f"Consulta: {report['fetched_at']}")
        if report.get("source_sync"):
            st.write(f"Sincronización más reciente comunicada por el proveedor: {report['source_sync']}")
        if report.get("coverage"):
            st.write("Calendarios consultados: "+", ".join(report["coverage"]))
        st.write(report.get("history_scope","Historial de la competición del próximo partido."))
        st.write(f"Ventana solicitada: {count} partidos. Muestra utilizable: {home} {len(frames[0])}; {away} {len(frames[1])}.")
        st.markdown("#### Cómo se calculan los escenarios")
        st.write("Goles, córners y tiros: promedio entre lo que produce un equipo y lo que concede su rival. "
                 "Amarillas y faltas: promedio del propio equipo. Se requieren al menos 3 observaciones por componente. "
                 "Las cantidades siguen distribuciones Poisson independientes. Los rangos son cuantiles 10%–90%; "
                 "no incluyen la incertidumbre de estimar las medias con pocos partidos.")
        st.write("Este modelo sencillo no está calibrado ni validado para estas nuevas estadísticas. "
                 "No incorpora lesiones, alineaciones, árbitro ni cambios de entrenador. La independencia y la dispersión "
                 "Poisson pueden ser poco realistas, especialmente para tarjetas y córners. No se atribuye a estos escenarios "
                 "el rendimiento de los modelos históricos del panel avanzado.")
        st.write("Sólo se incluyen partidos FT de 90 minutos anteriores al día actual UTC. No se incluyen prórrogas, "
                 "penaltis ni encuentros en curso. El filtro de localía reduce la muestra descargada y puede dejarla insuficiente.")
        st.write("Tarjetas: se usan totales explícitos del equipo o la suma de jugadores de una plantilla completa. "
                 "No se deducen ceros de un listado de eventos vacío. No se incluyen tarjetas del cuerpo técnico; "
                 "la codificación de segundas amarillas depende del proveedor.")
        if report.get("source_url"):
            st.link_button("Documentación de la fuente",report["source_url"])
        st.download_button("Descargar partidos utilizados (CSV)",report["history"].to_csv(index=False).encode("utf-8-sig"),
                           file_name="partidos_consultados.csv",mime="text/csv")


def welcome():
    html('''<div class="fi-welcome"><div class="fi-eyebrow">Una previa clara, en un solo lugar</div>
<h2>Busca un equipo. Conoce el enfrentamiento.</h2>
<p>Consulta el próximo partido registrado y compara la forma reciente de los dos equipos con datos publicados por la fuente. Empieza escribiendo Barcelona, Arsenal o el club que quieras analizar.</p>
<div class="fi-features"><div class="fi-feature">01 &nbsp; Próximo rival<small>Fecha, competición y condición de local o visitante.</small></div>
<div class="fi-feature">02 &nbsp; Ambos equipos<small>Goles, córners, tarjetas, tiros y posesión, con cobertura visible.</small></div>
<div class="fi-feature">03 &nbsp; Escenarios del partido<small>Probabilidades y cantidades estimadas, con sus límites.</small></div></div></div>''')


def main():
    st.set_page_config(page_title="Football Intelligence · Previa",page_icon="⚽",layout="wide")
    html("<style>"+Path(__file__).with_name("assets").joinpath("dashboard.css").read_text(encoding="utf-8")+"</style>")
    html('<div class="fi-brand"><span class="fi-mark">FI</span> FOOTBALL INTELLIGENCE <span class="fi-tag">MATCH CENTER · 05</span></div>')
    st.title("El próximo partido, en perspectiva.")
    html('<p class="fi-intro">Resultados del partido, comparación de equipos y protagonistas.</p>')
    settings,now=Settings(),datetime.now(timezone.utc)
    with st.sidebar:
        st.markdown("### Configura tu previa")
        source=st.radio("Fuente de datos",[SOURCES[0]] if settings.public_mode else SOURCES)
        count=st.select_slider("Partidos recientes",options=[5,10],value=5)
        venue_only=st.checkbox("Filtrar por local / visitante",help="Aplica el filtro a los partidos descargados. Puede reducir la muestra.")
        zone=ZONES[st.selectbox("Zona horaria",list(ZONES),index=1)]
        key=""
        if source==SOURCES[1] and not settings.public_mode:
            key=st.text_input("Clave API-Football",value=settings.api_football_key.get_secret_value(),type="password")
            st.caption("La clave no se guarda en archivos ni se muestra en errores.")
            st.link_button("Gestionar mi clave","https://dashboard.api-football.com/")
        elif source==SOURCES[0]:
            st.caption("Fuente comunitaria gratuita sin clave: worldcup26.ir. La cobertura y disponibilidad pueden variar.")
        st.divider()
        st.caption("Verde: local · Azul: visitante\n\nAnálisis estadístico. Las probabilidades son estimaciones, no recomendaciones de apuestas.")
    with st.form("search_team"):
        a,b,c=st.columns([2,4,1.6],vertical_alignment="bottom")
        league_name=a.selectbox("Liga",list(LEAGUES))
        query=b.text_input("Equipo",value="Barcelona",placeholder="Escribe el nombre del equipo…")
        submitted=c.form_submit_button("Analizar equipo",type="primary",width="stretch")
    league=LEAGUES[league_name]
    if source==SOURCES[2]:
        try:
            data=load_matches(connect(settings.database_url.get_secret_value()),simulated=False)
        except Exception:
            st.error("No se pudo leer la base local. Revisa FIA_DATABASE_URL.")
            return
        choices=[p for p in find_teams(data,query) if p[0]==league_name]
        if not choices:
            st.info("No hay coincidencias en el histórico local para esta liga y nombre.")
            return
        competition,team=st.selectbox("Equipo encontrado",choices,format_func=lambda p:p[1])
        frame=recent_matches(data,team,competition,now,count)
        st.warning("Archivo histórico: no identifica el próximo partido actual. Selecciona Gratis · sin clave para consultar el calendario.")
        show_team(team,frame)
        return
    if source==SOURCES[1] and not key.strip():
        st.info("Introduce tu clave en la barra lateral o selecciona Gratis · sin clave.")
        return
    client_key=(source,key,settings.public_mode)
    if st.session_state.get("client_key")!=client_key:
        st.session_state.client=(public_client() if settings.public_mode else CommunityFootball()) if source==SOURCES[0] else FootballAPI(key)
        st.session_state.client_key=client_key
    client=st.session_state.client
    context=(source,league)
    if submitted:
        try:
            with st.spinner("Buscando el equipo en el catálogo…"):
                choices=client.search(query,league) if source==SOURCES[0] else client.search(query)
            st.session_state.candidates=(context,choices)
            st.session_state.pop("match_report",None)
        except (ProviderError,ValueError,KeyError,TypeError) as exc:
            st.error(str(exc) if isinstance(exc,(ProviderError,ValueError)) else "El catálogo recibido está incompleto.")
            st.session_state.pop("candidates",None)
            return
    saved_candidates=st.session_state.get("candidates")
    if not saved_candidates or saved_candidates[0]!=context:
        welcome()
        return
    choices=saved_candidates[1]
    if not choices:
        st.info("No encontramos ese nombre en la liga seleccionada. Cambia la liga o prueba el nombre oficial.")
        return
    selected=st.selectbox("Equipo encontrado",choices,format_func=lambda t:f"{t['name']} · {t.get('country','')}") if len(choices)>1 else choices[0]
    request_key=(*context,selected["id"],count)
    saved=st.session_state.get("match_report")
    refresh=st.button("Actualizar consulta" if ((saved and saved[0]==request_key) or (submitted and len(choices)==1)) else "Ver enfrentamiento")
    if (submitted and len(choices)==1) or refresh:
        if refresh and not settings.public_mode:
            client.cache.clear()
        try:
            with st.spinner("Consultando el calendario y las estadísticas de ambos equipos…"):
                progress=st.progress(0.0)
                kwargs={"league":league} if source==SOURCES[0] else {}
                report=client.report(selected["id"],count=count,progress=progress.progress,**kwargs)
                progress.empty()
                st.session_state.match_report=(request_key,report)
        except (ProviderError,ValueError,KeyError,TypeError,IndexError) as exc:
            st.error(str(exc) if isinstance(exc,(ProviderError,ValueError)) else "El proveedor devolvió datos incompletos. Prueba más tarde.")
    saved=st.session_state.get("match_report")
    if saved and saved[0]==request_key:
        show_match_report(saved[1],count,venue_only,now,zone)
    elif not submitted:
        st.info("Pulsa Ver enfrentamiento para cargar la ventana seleccionada.")
    html('<div class="fi-footer">FOOTBALL INTELLIGENCE · Datos publicados por el proveedor · Estimaciones independientes de goles, córners y tarjetas · v0.5</div>')
