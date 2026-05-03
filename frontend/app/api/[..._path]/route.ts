import { NextRequest, NextResponse } from "next/server";

console.log("[API PROXY] route module loaded:", __filename);

export const runtime = "nodejs"; // recommended for SSE proxying
const DEBUG = { "x-route-hit": "api-proxy-router-ts" };

function getCorsHeaders() {
  return {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, PUT, PATCH, DELETE, OPTIONS",
    "Access-Control-Allow-Headers": "*",
  };
}

async function handleRequest(req: NextRequest, method: string) {
  console.log("[API PROXY] HIT", method, req.nextUrl.pathname);
  try {
    const baseUrl = process.env["LANGGRAPH_API_URL"]; // set to your FastAPI base, e.g. http://localhost:8000
    if (!baseUrl) {
      return NextResponse.json({ error: "LANGGRAPH_API_URL is not set" }, { status: 500, headers: DEBUG });
    }

    const path = req.nextUrl.pathname.replace(/^\/?api\//, "");
    const url = new URL(req.url);

    const searchParams = new URLSearchParams(url.search);
    searchParams.delete("_path");
    searchParams.delete("nxtP_path");
    const queryString = searchParams.toString() ? `?${searchParams.toString()}` : "";

    // Forward only what you need
    const headers: Record<string, string> = {};

    // Preserve content-type for JSON bodies
    const contentType = req.headers.get("content-type");
    if (contentType) headers["content-type"] = contentType;

    // IMPORTANT for SSE: accept text/event-stream
    const accept = req.headers.get("accept");
    if (accept) headers["accept"] = accept;

    const options: RequestInit = { method, headers };

    if (["POST", "PUT", "PATCH"].includes(method)) {
      options.body = await req.text();
    }

    const upstream = await fetch(`${baseUrl}/${path}${queryString}`, options);

    // SSE passthrough headers (avoid buffering)
    const passthroughHeaders: Record<string, string> = {
      ...getCorsHeaders(),
      "connection": "keep-alive",
      "x-accel-buffering": "no",
      "content-encoding": "identity",
    };

    const ct = upstream.headers.get("content-type");
    if (ct) passthroughHeaders["content-type"] = ct;

    const cacheControl = upstream.headers.get("cache-control");
    if (cacheControl) passthroughHeaders["cache-control"] = cacheControl;

    return new NextResponse(upstream.body, {
      status: upstream.status,
      statusText: upstream.statusText,
      headers: passthroughHeaders,
    });
  } catch (e: any) {
    return NextResponse.json({ error: e.message }, { status: e.status ?? 500, headers: DEBUG });
  }
}


export const GET = (req: NextRequest) => handleRequest(req, "GET");
export const POST = (req: NextRequest) => handleRequest(req, "POST");
export const PUT = (req: NextRequest) => handleRequest(req, "PUT");
export const PATCH = (req: NextRequest) => handleRequest(req, "PATCH");
export const DELETE = (req: NextRequest) => handleRequest(req, "DELETE");

export const OPTIONS = () =>
  new NextResponse(null, { status: 204, headers: { ...getCorsHeaders(), ...DEBUG } });
