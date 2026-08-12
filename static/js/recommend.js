/**
 * Recommended portfolio card: fetches /api/recommendation and renders the
 * suggested holdings, portfolio-level risk, changes since the last
 * recommendation, and the diff against the user's actual positions.
 */

document.addEventListener('DOMContentLoaded', function () {
    const area = document.getElementById('recommendation-area');
    const asOfEl = document.getElementById('rec-asof');
    const refreshButton = document.getElementById('refresh-recommendation-button');
    if (!area) return;

    const fmt = (v, d = 2) => (v === null || v === undefined || isNaN(v)) ? '–' : Number(v).toFixed(d);

    async function loadRecommendation() {
        try {
            const response = await fetch('/api/recommendation');
            if (!response.ok) {
                throw new Error(`Server error: ${response.statusText}`);
            }
            render(await response.json());
        } catch (error) {
            console.error('Recommendation load failed:', error);
            area.innerHTML = `<div class="alert alert-warning">Could not load recommendation: ${error.message}</div>`;
        }
    }

    function render(rec) {
        if (asOfEl && rec.asOf) {
            asOfEl.textContent = `(from scan at ${rec.asOf})`;
        }
        if (!rec.holdings || rec.holdings.length === 0) {
            area.innerHTML = `<div class="alert alert-info">${rec.note || 'Nothing to recommend right now.'}</div>`;
            return;
        }

        let html = `
            <p class="small text-muted mb-2">
                Top ${rec.holdings.length} of max ${rec.max_names} names by the ${rec.selection_horizon} composite score
                (the horizon validated in the backtest), weighted by inverse volatility, capped at 15% per name.
                Correlated near-duplicates are skipped and existing picks are kept unless they fall out of favor,
                so this should change slowly - it is not meant to be rebalanced every scan.${helpIcon('recommendation')}
            </p>`;

        const r = rec.risk;
        if (r && r.ann_vol !== undefined) {
            html += `
            <div class="small mb-3">
                <strong>Portfolio profile:</strong>
                <span title="${METRIC_HELP.beta}">beta ${fmt(r.beta)}</span> ·
                <span title="${METRIC_HELP.annVol}">vol ${fmt(r.ann_vol, 1)}%</span> ·
                <span title="${METRIC_HELP.varCvar}">VaR/CVaR ${fmt(r.var_95)}% / ${fmt(r.cvar_95)}%</span> ·
                <span title="${METRIC_HELP.effectivePositions}">effective positions ${fmt(r.effective_positions, 1)}</span> ·
                <span title="${METRIC_HELP.avgCorrelation}">avg corr ${fmt(r.avg_correlation)}</span>
            </div>`;
        }

        html += `
            <div class="table-responsive">
                <table class="table table-sm table-hover align-middle" id="recommendation-table">
                    <thead class="table-light">
                        <tr>
                            <th>Symbol</th>
                            <th title="${METRIC_HELP.recWeight}">Weight</th>
                            <th title="Composite score and watchlist rank at the ${rec.selection_horizon} horizon - the backtested selection signal.">Score (rank)</th>
                            <th title="${METRIC_HELP.recTiming}">Timing 1d / 1w</th>
                            <th title="${METRIC_HELP.sharpe}">Sharpe</th>
                            <th title="${METRIC_HELP.beta}">Beta</th>
                            <th>Sector</th>
                            <th>Signal</th>
                        </tr>
                    </thead>
                    <tbody>`;

        rec.holdings.forEach(h => {
            const timing = ['1d', '1w'].map(p => {
                const t = h.timing && h.timing[p];
                if (!t) return '–';
                const cls = t.score >= 15 ? 'text-success' : t.score <= -15 ? 'text-danger' : 'text-muted';
                return `<span class="${cls}">${fmt(t.score, 0)}</span>`;
            }).join(' / ');
            html += `
                <tr>
                    <td class="fw-bold">${h.symbol}</td>
                    <td>${fmt(h.weight, 1)}%</td>
                    <td>${fmt(h.score, 1)}${h.rank ? ` (${h.rank})` : ''}</td>
                    <td>${timing}</td>
                    <td>${fmt(h.sharpe)}</td>
                    <td>${fmt(h.beta)}</td>
                    <td class="small">${h.sector || '–'}</td>
                    <td class="small" title="${h.top_signal ? signalHelp(h.top_signal) : ''}">${h.top_signal || ''}</td>
                </tr>`;
        });
        html += `</tbody></table></div>`;

        const ch = rec.changes || {};
        if ((ch.added && ch.added.length) || (ch.dropped && ch.dropped.length)) {
            const bits = [];
            if (ch.added && ch.added.length) bits.push(`added ${ch.added.join(', ')}`);
            if (ch.dropped && ch.dropped.length) bits.push(`dropped ${ch.dropped.join(', ')}`);
            html += `<div class="small text-muted mb-2">Since last recommendation:${helpIcon('recChanges')} ${bits.join('; ')}</div>`;
        }

        const vs = rec.vs_current;
        if (vs && ((vs.not_held && vs.not_held.length) || (vs.held_not_recommended && vs.held_not_recommended.length))) {
            html += `<div class="small mb-1"><strong>Versus your portfolio:</strong>${helpIcon('recVsCurrent')}</div>`;
            if (vs.not_held && vs.not_held.length) {
                const items = vs.not_held.slice(0, 10)
                    .map(d => `${d.symbol} (${fmt(d.weight, 1)}%)`).join(', ');
                html += `<div class="small text-success mb-1">Recommended but not held: ${items}</div>`;
            }
            if (vs.held_not_recommended && vs.held_not_recommended.length) {
                const items = vs.held_not_recommended
                    .map(d => `${d.symbol} (${fmt(d.current_weight, 1)}% of portfolio)`).join(', ');
                html += `<div class="small text-danger mb-1">Held but not currently recommended: ${items}</div>`;
            }
        }

        html += `<div class="small text-muted mt-2">Research output, not investment advice.</div>`;

        area.innerHTML = html;
        if (typeof initTooltips === 'function') {
            initTooltips(area);
        }
    }

    if (refreshButton) {
        refreshButton.addEventListener('click', loadRecommendation);
    }

    // initial load pulls from the last stored scan
    setTimeout(loadRecommendation, 1500);

    window.loadRecommendation = loadRecommendation;
});
