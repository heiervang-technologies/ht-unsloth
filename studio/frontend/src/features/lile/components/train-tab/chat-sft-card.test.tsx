// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { describe, it, expect, spyOn, beforeEach } from "bun:test";
import { render, screen, cleanup, fireEvent } from "@testing-library/react";
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

    // wait for the async handler
    await Promise.resolve();

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
});
