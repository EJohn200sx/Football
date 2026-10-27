"""Player cards and rankings for the Streamlit match centre."""
from html import escape
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from football_ai.player_insights import PLAYER_EVENTS


def player_spotlights(rankings):
    cards=[]
    for key, title in [('goal','GOL'),('assist','ASISTENCIA'),('shot_on_target','TIRO AL ARCO')]:
        ranking=rankings.get(key,pd.DataFrame())
        if ranking.empty:
            cards.append(f'<div class="fi-player-card"><div class="fi-eyebrow">{title} · ≥1</div><h3>Sin muestra suficiente</h3><p>Se necesitan 3 apariciones con dato.</p></div>')
            continue
        r=ranking.iloc[0]
        cards.append(f'<div class="fi-player-card {r.side}"><div class="fi-eyebrow">{title} · ≥1</div>'
            f'<div class="fi-player-prob">{r.probability:.0%}</div><h3>{escape(r.player)}</h3><p>{escape(r.team)}</p>'
            f'<div class="fi-player-track"><span style="width:{r.probability*100:.2f}%"></span></div>'
            f'<small>Ocurrió en {int(r.hits)} de {int(r["sample"])} apariciones · si participa</small></div>')
    st.markdown('<div class="fi-section">Jugadores con mayor estimación disponible</div><div class="fi-player-grid">'+''.join(cards)+'</div>',unsafe_allow_html=True)
    st.caption('Condicionado a que participen en condiciones comparables. No es una alineación confirmada. '
               'Las estimaciones se calculan por separado y no suman 100%. Consulta la muestra en Jugadores.')


def show_players(report, rankings):
    st.subheader('Los protagonistas del encuentro')
    st.info('Estas estimaciones suponen que el jugador participa. La fuente no confirma aquí la próxima alineación, '
            'lesiones ni minutos previstos. Los candidatos proceden de la plantilla del último partido recibido.')
    if not any(not df.empty for df in rankings.values()):
        st.warning('No hay suficientes fichas de jugadores para elaborar la comparación. La sección necesita al menos '
                   'tres apariciones con esa estadística y una plantilla reciente disponible. El conector con clave no aporta jugadores en esta versión.')
        return
    player_spotlights(rankings)
    left,right=st.columns([2,1])
    key=left.selectbox('Acción del jugador',list(PLAYER_EVENTS),format_func=lambda k:PLAYER_EVENTS[k][0])
    names=[report['fixture']['home_team'],report['fixture']['away_team']]
    team=right.selectbox('Equipo de jugadores',['Ambos equipos',*names])
    ranking=rankings.get(key,pd.DataFrame())
    if team!='Ambos equipos' and not ranking.empty:
        ranking=ranking[ranking.team==team]
    if ranking.empty:
        st.info('No hay tres apariciones válidas para esa acción y selección de equipo.')
        return
    chart_data=ranking.head(12).iloc[::-1]
    fig=go.Figure(go.Bar(x=chart_data.probability*100,y=chart_data.player,orientation='h',
        marker_color=['#b5f36d' if s=='home' else '#73bcff' for s in chart_data.side],
        customdata=chart_data[['team','hits','sample']].to_numpy(),
        text=[f'{p:.0%}' for p in chart_data.probability],textposition='outside',cliponaxis=False,
        hovertemplate='%{y} · %{customdata[0]}<br>Estimación si participa: %{x:.1f}%<br>Ocurrió en %{customdata[1]}/%{customdata[2]} apariciones<extra></extra>'))
    fig.update_layout(template='plotly_dark',paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
        height=max(280,len(chart_data)*38),margin=dict(l=5,r=40,t=20,b=40),font=dict(color='#cad6e1'),
        xaxis=dict(range=[0,105],title='Estimación condicionada (%)',gridcolor='#293642'),yaxis=dict(automargin=True))
    st.plotly_chart(fig,width='stretch',config={'displayModeBar':False})
    table=pd.DataFrame([{'Jugador':r.player,'Equipo':r.team,'Estimación si participa':r.probability,
        'Ocurrió en':f'{r.hits}/{r.sample}','Media por aparición':round(r.mean,2),
        'Titular en':f'{r.starts}/{r.appearances}',
        'Incertidumbre 80%':f'{r.lower:.0%}–{r.upper:.0%}'} for r in ranking.itertuples()])
    st.dataframe(table,hide_index=True,width='stretch',column_config={
        'Estimación si participa':st.column_config.ProgressColumn(format='percent',min_value=0,max_value=1)})
    st.caption('El rango 80% describe incertidumbre de la frecuencia bajo el modelo Beta; no es una precisión demostrada. '
               'Pocas apariciones producen rangos amplios. No hay ajuste por minutos, posición, rival ni cambios de plantilla.')
    with st.expander('Cómo interpretar los porcentajes'):
        st.write('Para cada acción se cuenta en cuántas apariciones ocurrió. Se aplica el suavizado de Jeffreys: '
                 '(apariciones con la acción + 0.5) / (apariciones con dato + 1). Por ejemplo, 2 de 5 produce 41.7%. '
                 'Evita afirmar 0% o 100% con una muestra pequeña. No se ha calibrado esta proyección para partidos futuros.')
        st.write('Un suplente que no entró no cuenta como aparición. Un dato ausente no se transforma en cero. '
                 'No se inventan minutos ni estadísticas por 90 minutos. El filtro de localía de equipos no se aplica a los jugadores.')
    st.download_button('Descargar ranking de jugadores (CSV)',ranking.to_csv(index=False).encode('utf-8-sig'),
                       file_name=f'jugadores_{key}.csv',mime='text/csv')
