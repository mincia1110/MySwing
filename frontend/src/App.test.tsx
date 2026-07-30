import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it } from "vitest";
import { App } from "./App";
import { I18nProvider } from "./i18n";

const originalLocalStorage = Object.getOwnPropertyDescriptor(
  window,
  "localStorage",
);

function stubStoredLanguage(initial: string | null) {
  let stored = initial;
  Object.defineProperty(window, "localStorage", {
    configurable: true,
    value: {
      getItem: (key: string) => (key === "myswing.language" ? stored : null),
      setItem: (key: string, value: string) => {
        if (key === "myswing.language") stored = value;
      },
    },
  });
}

function renderApp() {
  return render(
    <I18nProvider>
      <MemoryRouter>
        <App />
      </MemoryRouter>
    </I18nProvider>,
  );
}

afterEach(() => {
  if (originalLocalStorage) {
    Object.defineProperty(window, "localStorage", originalLocalStorage);
  } else {
    Reflect.deleteProperty(window, "localStorage");
  }
  document.documentElement.lang = "";
});

describe("App language switching", () => {
  it("syncs document.documentElement.lang with the selected language", async () => {
    const user = userEvent.setup();
    stubStoredLanguage(null);
    renderApp();

    expect(document.documentElement.lang).toBe("ko");

    await user.click(screen.getByRole("button", { name: "English" }));
    expect(document.documentElement.lang).toBe("en");

    await user.click(screen.getByRole("button", { name: "한국어" }));
    expect(document.documentElement.lang).toBe("ko");
  });

  it("restores the stored language into document.lang on load", () => {
    stubStoredLanguage("en");
    renderApp();
    expect(document.documentElement.lang).toBe("en");
  });
});
