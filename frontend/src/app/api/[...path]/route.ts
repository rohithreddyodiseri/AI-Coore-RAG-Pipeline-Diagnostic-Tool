import { proxyBackend } from "@/lib/backend";

async function handle(request: Request, context: { params: Promise<{ path: string[] }> }) {
  const { path } = await context.params;
  return proxyBackend(request, path);
}

export { handle as GET, handle as POST };
