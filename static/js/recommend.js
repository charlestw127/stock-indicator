/**
 * Recommended portfolio card: fetches /api/recommendation and renders the
 * suggested holdings with dollar/share targets, portfolio-level risk,
 * changes since the last recommendation, and a concrete rebalance plan
 * against the user's actual positions.
 */

document.addEventListener('DOMContentLoaded', function () {
    const area = document.getElementById('recommendation-area');
    const asOfEl = document.getElementById('rec-asof');
    const refreshButton = document.getElementById('refresh-recommendation-button');
    if (!area) return;

    let currentBase = null; // user override, null = size off the portfolio

    const fmt = (v, d = 2) => (v === null || v === undefined || isNaN(v)) ? '–' : Number(v).toFixed(d);
    const money = (v) => (v === null || v === undefined || isNaN(v)) ? '–'
        : '$' + Number(v).toLocaleString(undefined, { maximumFractionDigits: 0 });

    async function loadRecommendation(base) {
        if (base !== undefined) currentBase = base;
        try {
            const url = currentBase ? `/api/recommendation?base=${currentBase}` : '/api/recommendation';
            const response = await fetch(url);
            if (!response.ok) {
                throw new Error(`Server error: ${response.statusText}`);
            }
            render(await response.json());
        } catch (error) {
            console.error('Recommendation load failed:', error);
            area.innerHTML = `<div class="alert alert-warning">Could not load recommendation: ${error.message}</div>`;
        }
    }

    /** Entry verdict from the tactical scores: is now a good moment to buy? */
    function timingCell(h) {
        const t = h.timing || {};
        const d = t['1d'] ? t['1d'].score : null;
        const w = t['1w'] ? t['1w'].score : null;
        const nums = `${fmt(d, 0)} / ${fmt(w, 0)}`;
        const v = (d !== null && d !== undefined) ? d : w;
        if (v === null || v === undefined) return nums;
        let label, cls;
        if (v >= 15) { label = 'enter'; cls = 'text-success fw-bold'; }
        else if (v <= -15) { label = 'wait'; cls = 'text-danger fw-bold'; }
        else { label = 'ok'; cls = 'text-muted'; }
        return `${nums} · <span class="${cls}">${label}</span>`;
    }

    function render(rec) {
        if (asOfEl && rec.asOf) {
            asOfEl.textContent = `(from scan at ${rec.asOf})`;
        }
        if (!rec.holdings || rec.holdings.length === 0) {
            area.innerHTML = `<div class="alert alert-info">${rec.note || 'Nothing to recommend right now.'}</div>`;
            return;
        }

        const rb = rec.rebalance;
        const hasTargets = rec.holdings.some(h => h.target_value !== undefined);

        let html = `
            <p class="small text-muted mb-2">
                Top ${rec.holdings.length} of max ${rec.max_names} names by the ${rec.selection_horizon} composite score
                (the horizon validated in the backtest), weighted by inverse volatility, capped at 15% per name.
                Correlated near-duplicates are skipped and existing picks are kept unless they fall out of favor,
                so this should change slowly - it is not meant to be rebalanced every scan.${helpIcon('recommendation')}
            </p>`;

        // dollar base control
        html += `
            <div class="d-flex align-items-center flex-wrap gap-2 mb-3">
                <div class="input-group input-group-sm" style="max-width: 240px;">
                    <span class="input-group-text">Base $</span>
                    <input type="number" class="form-control" id="rec-base-input" min="0" step="100"
                           value="${rb ? rb.base_value : ''}" placeholder="portfolio value">
                    <button class="btn btn-outline-secondary" id="rec-base-apply">Apply</button>
                </div>
                <span class="small text-muted">
                    ${rb && rb.from_portfolio
                        ? 'Sized off your portfolio market value' + helpIcon('recTarget')
                        : rb ? 'Sized off your custom base amount' + helpIcon('recTarget')
                             : 'Add portfolio positions or enter a base amount to get dollar targets'}
                </span>
            </div>`;

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
                            <th title="${METRIC_HELP.recWeight}">Weight</th>` +
            (hasTargets ? `
                            <th title="${METRIC_HELP.recTarget}">Target $</th>
                            <th title="${METRIC_HELP.recTarget}">Shares</th>` : '') + `
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
            html += `
                <tr>
                    <td class="fw-bold">${h.symbol}</td>
                    <td>${fmt(h.weight, 1)}%</td>` +
                (hasTargets ? `
                    <td>${money(h.target_value)}</td>
                    <td title="at $${fmt(h.price)} per share">${fmt(h.target_shares, 2)}</td>` : '') + `
                    <td>${fmt(h.score, 1)}${h.rank ? ` (${h.rank})` : ''}</td>
                    <td>${timingCell(h)}</td>
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

        if (rb && rb.trades && rb.trades.length > 0) {
            html += `
            <div class="mt-3">
                <div class="small mb-1">
                    <strong>Rebalance plan</strong>${helpIcon('recRebalance')}
                    <span class="text-muted">
                        base ${money(rb.base_value)} · buys ${money(rb.buy_total)} · sells ${money(rb.sell_total)}
                        ${rb.net_cash_needed > 0 ? `· needs ${money(rb.net_cash_needed)} new cash` : ''}
                        · trades under ${money(rb.min_trade)} skipped
                    </span>
                </div>
                <div class="table-responsive">
                    <table class="table table-sm align-middle">
                        <thead class="table-light">
                            <tr>
                                <th>Symbol</th><th>Action</th><th>Amount</th><th>Shares</th>
                                <th>Now → Target</th>
                            </tr>
                        </thead>
                        <tbody>`;
            rb.trades.forEach(t => {
                const cls = t.action === 'buy' ? 'text-success fw-bold' : 'text-danger fw-bold';
                html += `
                    <tr>
                        <td class="fw-bold">${t.symbol}</td>
                        <td class="${cls}">${t.action.toUpperCase()}</td>
                        <td>${money(Math.abs(t.delta_value))}</td>
                        <td title="at $${fmt(t.price)} per share">${fmt(Math.abs(t.delta_shares), 2)}</td>
                        <td class="small text-muted">${fmt(t.current_weight, 1)}% → ${fmt(t.target_weight, 1)}%</td>
                    </tr>`;
            });
            html += `</tbody></table></div></div>`;
        } else if (rb) {
            html += `<div class="small text-success mt-2">Your holdings are already within 1% of the recommended weights - nothing to trade.</div>`;
        }

        html += `<div class="small text-muted mt-2">Research output, not investment advice. Amounts use the latest cached prices.</div>`;

        area.innerHTML = html;
        if (typeof initTooltips === 'function') {
            initTooltips(area);
        }

        const baseInput = document.getElementById('rec-base-input');
        const baseApply = document.getElementById('rec-base-apply');
        if (baseApply && baseInput) {
            const apply = () => {
                const v = parseFloat(baseInput.value);
                loadRecommendation((!isNaN(v) && v > 0) ? v : null);
            };
            baseApply.addEventListener('click', apply);
            baseInput.addEventListener('keydown', e => { if (e.key === 'Enter') apply(); });
        }
    }

    if (refreshButton) {
        refreshButton.addEventListener('click', () => loadRecommendation());
    }

    // initial load pulls from the last stored scan
    setTimeout(() => loadRecommendation(), 1500);

    window.loadRecommendation = loadRecommendation;
});
