import { createContext, useContext, useEffect, useState } from "react";

type Theme = "system" | "latte" | "frappe" | "macchiato" | "mocha";

interface ThemeContextValue {
  theme: Theme;
  resolvedTheme: string;
  setTheme: (theme: Theme) => void;
}

const ThemeContext = createContext<ThemeContextValue | null>(null);

function resolveTheme(pref: Theme): string {
  if (pref === "system") {
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "frappe" : "latte";
  }
  return pref;
}

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [theme, setTheme] = useState<Theme>(() => {
    const stored = localStorage.getItem("theme");
    if (stored && ["system", "latte", "frappe", "macchiato", "mocha"].includes(stored)) {
      return stored as Theme;
    }
    return "system";
  });

  const [resolvedTheme, setResolvedTheme] = useState(() => resolveTheme(theme));

  useEffect(() => {
    const applyTheme = () => {
      const resolved = resolveTheme(theme);
      setResolvedTheme(resolved);
      const html = document.documentElement;
      html.classList.remove("latte", "frappe", "macchiato", "mocha");
      html.classList.add(resolved);
    };

    applyTheme();
    localStorage.setItem("theme", theme);

    if (theme === "system") {
      const mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");
      mediaQuery.addEventListener("change", applyTheme);
      return () => mediaQuery.removeEventListener("change", applyTheme);
    }
  }, [theme]);

  return (
    <ThemeContext.Provider value={{ theme, resolvedTheme, setTheme }}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error("useTheme must be used within a ThemeProvider");
  }
  return context;
}
