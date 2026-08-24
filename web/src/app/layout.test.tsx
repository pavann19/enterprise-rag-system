import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import RootLayout, { metadata } from "./layout";

describe("RootLayout", () => {
  it("renders its children", () => {
    render(
      <RootLayout>
        <p>page content</p>
      </RootLayout>,
    );
    expect(screen.getByText("page content")).toBeInTheDocument();
  });

  it("exports metadata with a title and description", () => {
    expect(metadata.title).toBe("Enterprise RAG");
    expect(metadata.description).toMatch(/grounded strictly/i);
  });
});
