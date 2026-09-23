const CDP = process.env.CDP_URL || 'http://127.0.0.1:9222';
const APP = 'http://127.0.0.1:5173/';

async function json(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}: ${url}`);
  return response.json();
}

async function openTab() {
  const target = await json(`${CDP}/json/new?${encodeURIComponent(APP)}`, { method: 'PUT' });
  const ws = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => {
    ws.addEventListener('open', resolve, { once: true });
    ws.addEventListener('error', reject, { once: true });
  });

  let id = 0;
  const pending = new Map();
  const events = [];
  ws.addEventListener('message', (message) => {
    const payload = JSON.parse(message.data);
    if (payload.id && pending.has(payload.id)) {
      pending.get(payload.id)(payload);
      pending.delete(payload.id);
    } else if (payload.method) {
      events.push(payload);
    }
  });

  const send = (method, params = {}) => new Promise((resolve) => {
    const msg = { id: ++id, method, params };
    pending.set(msg.id, resolve);
    ws.send(JSON.stringify(msg));
  });

  await send('Runtime.enable');
  await send('Page.enable');
  await send('Network.enable');
  return { ws, send, events };
}

async function wait(ms) {
  await new Promise((resolve) => setTimeout(resolve, ms));
}

async function waitFor(send, expression, timeout = 8000) {
  const start = Date.now();
  while (Date.now() - start < timeout) {
    const value = await evalText(send, expression);
    if (value) return value;
    await wait(250);
  }
  return null;
}

async function evalText(send, expression) {
  const result = await send('Runtime.evaluate', {
    expression,
    awaitPromise: true,
    returnByValue: true,
  });
  return result.result?.result?.value;
}

async function main() {
  const { ws, send, events } = await openTab();
  const pages = [
    ['dashboard', `${APP}#dashboard`],
    ['vessel', `${APP}#vessel-intelligence`],
    ['idle', `${APP}#idle-intelligence`],
  ];

  const snapshots = {};
  const urls = {};
  for (const [name, url] of pages) {
    await send('Page.navigate', { url });
    await waitFor(send, `document.readyState === 'complete'`);
    await wait(2500);
    urls[name] = await evalText(send, `location.href`);
    snapshots[name] = await evalText(send, `document.body.innerText.slice(0, 6000)`);
  }

  await send('Page.navigate', { url: `${APP}#dashboard` });
  await waitFor(send, `document.readyState === 'complete'`);
  await wait(1500);
  await evalText(send, `
    [...document.querySelectorAll('button')]
      .find((button) => button.innerText.trim() === '1D')
      ?.click();
    true;
  `);
  await wait(2500);
  const promotedDashboard = await evalText(send, `document.body.innerText.slice(0, 7000)`);

  const consoleIssues = events
    .filter((event) => event.method === 'Runtime.consoleAPICalled')
    .filter((event) => ['error', 'warning'].includes(event.params.type))
    .map((event) => ({
      type: event.params.type,
      text: event.params.args.map((arg) => arg.value || arg.description || '').join(' '),
    }));

  const failedRequests = events
    .filter((event) => event.method === 'Network.loadingFailed')
    .map((event) => ({
      requestId: event.params.requestId,
      errorText: event.params.errorText,
      canceled: event.params.canceled,
    }));

  const apiResponses = events
    .filter((event) => event.method === 'Network.responseReceived')
    .map((event) => event.params.response)
    .filter((response) => response.url.includes('127.0.0.1:8000'))
    .map((response) => ({ url: response.url, status: response.status }));

  const checks = {
    dashboardHas1D: snapshots.dashboard.includes('1D'),
    dashboardShowsCoverage: snapshots.dashboard.includes('COVERED') || snapshots.dashboard.includes('HIGH') || snapshots.dashboard.includes('FALLBACK'),
    dashboardShowsFallback: snapshots.dashboard.includes('FLEXIBLE') && snapshots.dashboard.includes('Insufficient directional confidence'),
    promotedShowsCovered: promotedDashboard.includes('COVERED') || promotedDashboard.includes('HIGH'),
    promotedShowsBuyOrWait: promotedDashboard.includes('BUY NOW') || promotedDashboard.includes('WAIT'),
    promotedShowsFlexible: promotedDashboard.includes('FLEXIBLE'),
    vesselRendered: snapshots.vessel.includes('VESSEL INTELLIGENCE') && snapshots.vessel.includes('FLEET COMPOSITION'),
    idleRendered: snapshots.idle.includes('IDLE &') && snapshots.idle.includes('LIVE /idle-risk DATA'),
  };

  console.log(JSON.stringify({
    urls,
    previews: Object.fromEntries(Object.entries(snapshots).map(([key, value]) => [key, value.slice(0, 500)])),
    promotedDashboardPreview: promotedDashboard.slice(0, 1200),
    checks,
    apiResponses,
    consoleIssues,
    failedRequests,
  }, null, 2));
  ws.close();
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
