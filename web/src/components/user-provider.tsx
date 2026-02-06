import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import { type User, getUserToken, setUserToken, validateUser } from "../api";

interface UserContextType {
  user: User | null;
  loading: boolean;
  error: string | null;
}

const UserContext = createContext<UserContextType>({
  user: null,
  loading: true,
  error: null,
});

export function useUser() {
  return useContext(UserContext);
}

export function UserProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function initUser() {
      // Check URL params for user token
      const params = new URLSearchParams(window.location.search);
      const urlToken = params.get("user");

      // Use URL token if present, otherwise check localStorage
      const token = urlToken || getUserToken();

      if (!token) {
        setLoading(false);
        return;
      }

      try {
        const validatedUser = await validateUser(token);
        setUser(validatedUser);
        // Persist to localStorage
        setUserToken(token);

        // Remove token from URL (clean up) if it was in URL
        if (urlToken) {
          params.delete("user");
          const newUrl = params.toString()
            ? `${window.location.pathname}?${params.toString()}`
            : window.location.pathname;
          window.history.replaceState({}, "", newUrl);
        }
      }
      catch {
        setError("Invalid user token");
      }
      finally {
        setLoading(false);
      }
    }

    void initUser();
  }, []);

  return (
    <UserContext.Provider value={{ user, loading, error }}>
      {children}
    </UserContext.Provider>
  );
}
