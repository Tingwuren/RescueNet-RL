const resolveOrigin = (port) => {
  if (typeof window === "undefined") {
    return `http://localhost:${port}`;
  }
  const protocol = window.location.protocol || "http:";
  const hostname = window.location.hostname || "localhost";
  return `${protocol}//${hostname}:${port}`;
};

export const rescueApiBase =
  import.meta.env.VITE_API_BASE || `${resolveOrigin(8000)}/api`;

export const ns3ApiBase =
  import.meta.env.VITE_NS3_API_BASE || `${resolveOrigin(8001)}/api`;

export const ns3WebBase =
  import.meta.env.VITE_NS3_WEB_BASE || `${resolveOrigin(8080)}/index.html`;
