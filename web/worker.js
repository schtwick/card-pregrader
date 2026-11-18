// Runs the Python grading engine (Pyodide) off the main thread so the page stays responsive.
importScripts("./pyodide/pyodide.js");

let api = null;
let pyodide = null;

async function boot() {
  postMessage({ type: "status", text: "Starting the grading engine…" });
  pyodide = await loadPyodide({ indexURL: "./pyodide/" });
  postMessage({ type: "status", text: "Loading image tools (first time only takes a while)…" });
  await pyodide.loadPackage(["numpy", "opencv-python", "pillow"]);
  postMessage({ type: "status", text: "Loading the grader…" });
  const zip = await (await fetch("./py/pregrader.zip")).arrayBuffer();
  pyodide.unpackArchive(zip, "zip", { extractDir: "/home/pyodide" });
  pyodide.runPython("import sys; sys.path.insert(0, '/home/pyodide')");
  api = pyodide.pyimport("pregrader.webapi");
  postMessage({ type: "ready" });
}

const ready = boot().catch((e) => postMessage({ type: "fatal", text: String(e && e.message || e) }));

function toJs(v) {
  if (v && typeof v.toJs === "function") {
    const out = v.toJs({ dict_converter: Object.fromEntries, create_pyproxies: false });
    v.destroy();
    return out;
  }
  return v;
}

onmessage = async (ev) => {
  const { id, fn, args } = ev.data;
  try {
    await ready;
    if (!api) throw new Error("The grading engine didn't start.");
    const pyArgs = args.map((a) => (a instanceof Uint8Array ? pyodide.toPy(a) : (a && typeof a === "object" ? pyodide.toPy(a) : a)));
    const result = toJs(api[fn](...pyArgs));
    pyArgs.forEach((a) => a && typeof a.destroy === "function" && a.destroy());
    postMessage({ id, ok: true, result });
  } catch (e) {
    let msg = String(e && e.message || e);
    const m = msg.match(/(\w+Error: .*)\s*$/);   // last line of a Python traceback
    if (m) msg = m[1];
    postMessage({ id, ok: false, error: msg });
  }
};
