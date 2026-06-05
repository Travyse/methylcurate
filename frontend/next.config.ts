import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  serverExternalPackages: ["undici"],
  serverTimeout: 0,
  turbopack: {
    root: __dirname,
  },
};

export default nextConfig;