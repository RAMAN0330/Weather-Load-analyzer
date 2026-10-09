/**
 * forecastWorker.js
 * Web Worker: polls forecast job status and posts results back to main thread.
 * Messages in:  { type: 'start', jobId, apiBase, token }  |  { type: 'stop' }
 * Messages out: { type: 'progress', status, events }
 *               { type: 'result',   data }
 *               { type: 'error',    message }
 */

let _interval = null;
let _notFoundCount = 0;
const _NOT_FOUND_LIMIT = 8; // stop after 8 consecutive 404s (~24s); server restart detection

const joinApi = (base, path) =>
  String(base || '').replace(/\/$/, '') + (path.startsWith('/') ? path : '/' + path);

self.onmessage = function (e) {
  const { type, jobId, apiBase, token } = e.data;
  const fetchOpts = token ? { headers: { Authorization: 'Bearer ' + token } } : {};

  if (type === 'start') {
    if (_interval) clearInterval(_interval);
    _notFoundCount = 0;

    _interval = setInterval(async function () {
      try {
        const statusRes = await fetch(joinApi(apiBase, '/v2/forecast/job/' + jobId), fetchOpts);
        if (!statusRes.ok) {
          if (statusRes.status === 404) {
            _notFoundCount++;
            if (_notFoundCount >= _NOT_FOUND_LIMIT) {
              clearInterval(_interval);
              _interval = null;
              self.postMessage({
                type: 'error',
                message: 'Job expired — server may have restarted. Please re-run forecast.',
              });
            }
          }
          return;
        }
        _notFoundCount = 0; // reset on any successful response
        const job = await statusRes.json();

        self.postMessage({
          type: 'progress',
          status: job.status,
          elapsed: job.elapsed,
          events: Array.isArray(job.progress) ? job.progress : [],
        });

        if (job.status === 'done') {
          clearInterval(_interval);
          _interval = null;
          const resultRes = await fetch(
            joinApi(apiBase, '/v2/forecast/job/' + jobId + '/result'),
            fetchOpts
          );
          if (!resultRes.ok) {
            self.postMessage({ type: 'error', message: 'Failed to fetch result' });
            return;
          }
          const data = await resultRes.json();
          self.postMessage({ type: 'result', data });
        } else if (job.status === 'error') {
          clearInterval(_interval);
          _interval = null;
          self.postMessage({ type: 'error', message: job.error || 'Forecast failed' });
        }
      } catch (err) {
        // Transient network error — keep polling
      }
    }, 3000);
  }

  if (type === 'stop') {
    if (_interval) {
      clearInterval(_interval);
      _interval = null;
    }
  }
};
