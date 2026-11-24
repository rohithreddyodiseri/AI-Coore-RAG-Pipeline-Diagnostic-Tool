export async function POST(request: Request) {
  const baseUrl = process.env.RAG_API_BASE_URL;
  const token = process.env.RAG_REGISTRY_TOKEN;

  if (!baseUrl || !token) {
    return Response.json(
      { message: "Backend connection settings are missing." },
      { status: 500 }
    );
  }

  let body: unknown;

  try {
    body = await request.json();
  } catch {
    return Response.json(
      { message: "Invalid registration data." },
      { status: 400 }
    );
  }

  try {
    const response = await fetch(`${baseUrl}/api/v1/apps`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${token}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
      cache: "no-store",
      signal: AbortSignal.timeout(10000),
    });

    const data = await response.json();

    return Response.json(data, {
      status: response.status,
      headers: { "Cache-Control": "no-store" },
    });
  } catch {
    return Response.json(
      {
        message:
          "Cannot reach the Python backend. Check that it is running.",
      },
      { status: 502 }
    );
  }
}