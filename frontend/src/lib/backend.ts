/** Server-only gateway. Registry credentials never enter browser bundles. */
export async function proxyBackend(request: Request, path: string[]) {
  const baseUrl = process.env.RAG_API_BASE_URL ?? "http://127.0.0.1:4318";
  const token = process.env.RAG_REGISTRY_TOKEN;
  const accuracyRequest = path[0] === "accuracy-runs" && path.length === 1;
  const allowed =
    (path[0] === "accuracy-runs" && (path.length === 1 || path.length === 2)) ||
    (path.length === 1 && ["apps", "traces"].includes(path[0])) ||
    (path[0] === "apps" && path.length === 2 && /^[a-f0-9]{32}$/.test(path[1])) ||
    (path[0] === "analytics" && path[1] === "health" && path.length === 2) ||
    (path[0] === "traces" &&
      (path.length === 2 ||
        (path.length === 3 && ["spans", "diagnosis"].includes(path[2])) ||
        (path.length === 4 && path[2] === "artifacts")));

  if (!allowed) return Response.json({ message: "Route not found." }, { status: 404 });
  const diagnosisRequest = path[0] === "traces" && path.length === 3 && path[2] === "diagnosis";
  if (request.method !== "GET" && !(request.method === "POST" && (path.join("/") === "apps" || diagnosisRequest || accuracyRequest))) {
    return Response.json({ message: "Method not allowed." }, { status: 405 });
  }
  if (!token) {
    return Response.json(
      { message: "Set RAG_REGISTRY_TOKEN on the frontend server, or start with python scripts/dev.py." },
      { status: 503 },
    );
  }

  let body: string | undefined;
  if (request.method === "POST" && path.join("/") === "apps") {
    try {
      body = JSON.stringify(await request.json());
    } catch {
      return Response.json({ message: "Invalid registration data." }, { status: 400 });
    }
  }

  try {
    const url = new URL(`${baseUrl.replace(/\/$/, "")}/api/v1/${path.map(encodeURIComponent).join("/")}`);
    url.search = new URL(request.url).search;
    const response = await fetch(url, {
      method: request.method,
      headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
      body,
      cache: "no-store",
      redirect: "error",
      signal: AbortSignal.timeout(10000),
    });
    const data: unknown = await response.json();
    return Response.json(data, {
      status: response.status,
      headers: { "Cache-Control": "no-store" },
    });
  } catch {
    return Response.json(
      { message: "Cannot reach the Python backend. Check RAG_API_BASE_URL and that the backend is running." },
      { status: 502, headers: { "Cache-Control": "no-store" } },
    );
  }
}
