// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { describe, it, expect, spyOn, beforeEach } from "bun:test";
import { render, screen, cleanup, fireEvent, waitFor } from "@testing-library/react";
import { lileClient } from "../../api/lile-client";
import { ChatSftCard } from "./chat-sft-card";

describe("ChatSftCard", () => {
  beforeEach(() => {
    cleanup();
  });

  it("posts sft train body with user + assistant messages", async () => {
    const spy = spyOn(lileClient, "postTrain").mockImplementation(() =>
      Promise.resolve({ queued: true }),
    );

    render(<ChatSftCard />);

    const userTextarea = screen.getByLabelText(/user/i);
    const assistantTextarea = screen.getByLabelText(/assistant/i);

    fireEvent.change(userTextarea, { target: { value: "hi" } });
    fireEvent.change(assistantTextarea, { target: { value: "hello" } });

    const trainBtn = screen.getByRole("button", { name: /train/i });
    fireEvent.click(trainBtn);

    await waitFor(() => expect(spy.mock.calls.length).toBe(1));

    expect(spy.mock.calls[0][0]).toMatchObject({
      objective: "sft",
      samples: [
        {
          messages: [
            { role: "user", content: "hi" },
            { role: "assistant", content: "hello" },
          ],
        },
      ],
    });

    spy.mockRestore();
  });

  it("switches to weighted_sft when strength is not 1.0", async () => {
    const spy = spyOn(lileClient, "postTrain").mockImplementation(() =>
      Promise.resolve({ queued: true }),
    );

    render(<ChatSftCard />);

    fireEvent.change(screen.getByLabelText(/user/i), { target: { value: "hi" } });
    fireEvent.change(screen.getByLabelText(/assistant/i), { target: { value: "hello" } });
    fireEvent.change(screen.getByLabelText(/strength/i), { target: { value: "3" } });

    fireEvent.click(screen.getByRole("button", { name: /train/i }));

    await waitFor(() => expect(spy.mock.calls.length).toBe(1));

    expect(spy.mock.calls[0][0]).toMatchObject({
      objective: "weighted_sft",
      samples: [
        {
          weight: 3,
          messages: [
            { role: "user", content: "hi" },
            { role: "assistant", content: "hello" },
          ],
        },
      ],
    });

    spy.mockRestore();
  });
});
