import { proxyBackend } from "@/lib/backend";

export function GET(request: Request) {
  return proxyBackend(request, ["apps"]);
}

export function POST(request: Request) {
  return proxyBackend(request, ["apps"]);
}
