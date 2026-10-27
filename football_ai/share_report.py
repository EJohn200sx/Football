"""Portable HTML snapshots: no credentials, scripts, external assets or live calls."""
from __future__ import annotations
from html import escape
import pandas as pd
from football_ai.match_forecast import count_scenarios, score_scenarios
from football_ai.player_insights import PLAYER_EVENTS
from football_ai.team_insights import STAT_NAMES


def table(headers, rows):
    head = ''.join(f'<th>{escape(str(x))}</th>' for x in headers)
    body = ''.join('<tr>'+''.join(f'<td>{escape(str(x))}</td>' for x in row)+'</tr>' for row in rows)
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def export_html(report: dict, estimates: dict, summaries: list, rankings: dict,
                zone: str = "America/Lima", venue_only: bool = False) -> str:
    f = report['fixture']; home, away = f['home_team'], f['away_team']
    date = pd.to_datetime(f['date'], utc=True).tz_convert(zone)
    scores = score_scenarios(estimates.get('home_goals'), estimates.get('away_goals'))
    blocks = [f'<span class="eyebrow">FOOTBALL INTELLIGENCE · INFORME GUARDADO</span><h1>{escape(home)} <span>vs</span> {escape(away)}</h1>',
              f'<p>{escape(f["competition"])} · {date:%d/%m/%Y %H:%M} · {escape(zone)}</p>',
              '<p class="notice">Esta copia no se actualiza automáticamente. Estimaciones experimentales, no resultados garantizados ni recomendaciones de apuestas.</p>']
    if scores:
        blocks.append('<h2>Resultado del partido · modelo Poisson</h2><div class="probabilities">'+''.join(
            f'<div class="box"><small>{escape(name)}</small><strong>{p:.1%}</strong></div>'
            for name,p in [(home,scores['home_win']),('Empate',scores['draw']),(away,scores['away_win'])])+'</div>')
        blocks.append(table(['Marcador local–visitante','Probabilidad'],[
            [f"{s['home']}–{s['away']}",f"{s['probability']:.1%}"] for s in scores['scores']]+[
            ['Otros marcadores',f"{1-sum(s['probability'] for s in scores['scores']):.1%}"]]))
    scenarios = count_scenarios(estimates)
    if scenarios:
        blocks.append('<h2>Cantidades estimadas</h2>'+table(['Estadística',home,away,'Total','Rango del modelo ≥80%'],[
            [s['label'],f"{s['home']:.1f}",f"{s['away']:.1f}",f"{s['total']:.1f}",f"{s['low']}–{s['high']}"] for s in scenarios.values()]))
    blocks.append('<h2>Jugadores · si participan</h2><p>Basado en apariciones recientes comparables. No confirma alineación, minutos ni estado de lesiones. Las probabilidades de varios jugadores no suman 100%.</p>')
    for key in ['goal','assist','shot_on_target']:
        ranking = rankings.get(key,pd.DataFrame())
        if ranking.empty:
            continue
        chosen = pd.concat([ranking[ranking.side == side].head(5) for side in ['home','away']])
        blocks.append(f'<h3>{escape(PLAYER_EVENTS[key][0])}</h3>'+table(['Jugador','Equipo','Estimación','Ocurrió en','Rango de incertidumbre 80%'],[
            [r.player,r.team,f'{r.probability:.1%}',f'{r.hits}/{r.sample}',f'{r.lower:.0%}–{r.upper:.0%}'] for r in chosen.itertuples()]))
    if not any(not r.empty for r in rankings.values()):
        blocks.append('<p>No hay suficientes estadísticas de jugadores en esta consulta.</p>')
    blocks.append('<h2>Promedios de ambos equipos</h2>'+table(['Estadística',home,away,'Cobertura local / visitante'],[
        [STAT_NAMES[key],*(f"{s['metrics'][key]['mean']:.2f}" if s['metrics'][key]['mean'] is not None else '—' for s in summaries),
         ' / '.join(f"{s['metrics'][key]['available']}/{s['metrics'][key]['total']}" for s in summaries)]
        for key in ['goals_for','goals_against','corners','yellow_cards','red_cards','shots','shots_on_target','possession','xg']]))
    blocks.append('<h2>Procedencia y límites</h2>'+f'<p>Fuente: {escape(report.get("source","API-Football"))}. Consulta: {escape(report["fetched_at"])}.</p>'
        +f'<p>{escape(report.get("scope","Calendario sujeto a la cobertura del proveedor."))}</p>'
        +f'<p>Calendarios: {escape(", ".join(report.get("coverage",[])) or f["competition"])}. '
        +f'Muestra de equipos: {summaries[0]["n"]} / {summaries[1]["n"]} partidos. Filtro de localía: {"sí" if venue_only else "no"}.</p>'
        +'<p>Equipos: medias recientes y Poisson independiente, sin calibración. Jugadores: frecuencia suavizada de Jeffreys, (partidos con el evento + 0.5)/(apariciones válidas + 1). '
         'Rango de jugadores: cuantiles 10%–90% de una distribución Beta, incertidumbre del parámetro bajo supuestos de apariciones comparables. '
         'No se confunde con el rango de cantidades del partido. Sólo candidatos listados en el último encuentro recibido, con al menos tres apariciones con dato. '
         'Sin ajuste de minutos o rival. xG sólo cuando existe en el proveedor.</p>')
    css='''body{margin:0;background:#10151b;color:#e8eff6;font:15px/1.6 Arial,sans-serif}main{max-width:1000px;margin:0 auto;padding:42px 24px}h1{font-size:38px;line-height:1.2;margin:16px 0}h1 span{font-size:19px;color:#8fa2b4}h2{margin-top:38px;font-size:23px}h3{font-size:17px}.eyebrow{color:#b5f36d;font-weight:bold;letter-spacing:.1em;font-size:11px}p{color:#b2c1ce}.notice{border-left:3px solid #b5f36d;padding:12px 18px;background:#19232d}.probabilities{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.box{padding:20px;background:#19232d;border:1px solid #33434e;border-radius:14px}.box strong{display:block;font-size:34px;color:#b5f36d}.scroll{overflow:auto}table{width:100%;border-collapse:collapse;margin:20px 0;font-size:13px}th,td{text-align:left;padding:12px;border-bottom:1px solid #30404c}th{color:#b5f36d;background:#19232d}footer{margin-top:40px;color:#899cae;font-size:12px}@media(max-width:600px){h1{font-size:29px}.box{padding:12px}.box strong{font-size:25px}}@media print{body{background:white;color:#17232f}p,footer{color:#34475a}.box,th,.notice{background:#edf3ed}th,.box strong,.eyebrow{color:#365f20}table{font-size:11px}main{padding:0}.box,table{break-inside:avoid}}'''
    return '<!doctype html><html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'+f'<title>{escape(home)} vs {escape(away)} · Football Intelligence</title><style>{css}</style></head><body><main>'+''.join(blocks)+'<footer>Football Intelligence · v0.5 · Informe estático. Puedes imprimirlo o guardarlo como PDF desde el navegador.</footer></main></body></html>'
