// src/lib/api.ts

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:3000/api";

interface FetchOptions extends RequestInit {
  data?: unknown;
}

export async function apiFetch<T>(endpoint: string, options: FetchOptions = {}): Promise<T> {
  const { data, headers: customHeaders, ...restOptions } = options;

  // Automatically grab credentials
  const token = typeof window !== "undefined" ? localStorage.getItem("token") : null;
  const companyId = typeof window !== "undefined" ? localStorage.getItem("companyId") : null;

  // Build the standard headers
  const headers = new Headers(customHeaders);
  
  if (token) {
    headers.set("Authorization", `Bearer ${token}`);
  }
  
  if (companyId) {
    headers.set("X-Company-Id", companyId);
  }

  // Handle bodies (JSON or FormData for file uploads)
  if (data !== undefined) {
    // Native FormData check (no 'any' needed)
    if (data instanceof FormData) {
      // DO NOT set Content-Type; the browser handles it and sets the boundary automatically
      restOptions.body = data;
    } else {
      headers.set("Content-Type", "application/json");
      restOptions.body = JSON.stringify(data);
    }
  }

  // Execute the fetch
  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    headers,
    ...restOptions,
  });

  // Standardized Error Handling
  if (!response.ok) {
    // Only auto-redirect if NOT already on an auth page to prevent loops
    const isAuthPage = typeof window !== "undefined" && 
    (window.location.pathname === "/signin" || window.location.pathname === "/signup");

    if (response.status === 401 && typeof window !== "undefined" && !isAuthPage) {
      localStorage.removeItem("token");
      localStorage.removeItem("companyId");
      window.location.href = "/signin";
    }

    let errorMessage = `HTTP Error ${response.status}`;
    try {
      // Using 'unknown' instead of 'any' to satisfy ESLint
      const errorData = (await response.json()) as Record<string, unknown>;
      
      // Type-safe checking for standard Express and FastAPI errors
      if (errorData.message) {
        errorMessage = String(errorData.message);
      } else if (errorData.error) {
        errorMessage = String(errorData.error);
      } else if (errorData.detail) {
        errorMessage = typeof errorData.detail === 'string' 
          ? errorData.detail 
          : JSON.stringify(errorData.detail);
      }
    } catch {
      errorMessage = response.statusText || errorMessage;
    }
    
    throw new Error(errorMessage);
  }

  // Return the parsed JSON
  if (response.status === 204) {
    return {} as T;
  }

  return response.json();
}