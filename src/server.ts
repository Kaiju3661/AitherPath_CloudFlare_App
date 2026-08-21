import { createWorkersAI } from "workers-ai-provider";
import { callable, routeAgentRequest, type Schedule } from "agents";
import { getSchedulePrompt, scheduleSchema } from "agents/schedule";
import { AIChatAgent, type OnChatMessageOptions } from "@cloudflare/ai-chat";
import {
  convertToModelMessages,
  pruneMessages,
  stepCountIs,
  streamText,
  tool
} from "ai";
import { z } from "zod";
import { getAgentByName } from "agents";

interface JoobleJob {
  title: string;
  company: string;
  location: string;
  salary?: string;
  link: string;
}

interface JoobleResponse {
  totalCount: number;
  jobs: JoobleJob[];
}

interface ChatState {
  googleTokens?: {
    accessToken: string;
    refreshToken?: string;
    expiresAt: number;
  };
}

export class ChatAgent extends AIChatAgent<Env, ChatState> {
  maxPersistedMessages = 100;
  chatRecovery = true;
  // Wait for MCP connections to be re-established after hibernation before
  // processing a message, so MCP tools aren't intermittently missing.
  waitForMcpConnections = true;

  initialState: ChatState = {};
  
  onStart() {
    // Configure OAuth popup behavior for MCP servers that require authentication
    this.mcp.configureOAuthCallback({
      customHandler: (result) => {
        if (result.authSuccess) {
          return new Response("<script>window.close();</script>", {
            headers: { "content-type": "text/html" },
            status: 200
          });
        }
        return new Response(
          `Authentication Failed: ${result.authError || "Unknown error"}`,
          { headers: { "content-type": "text/plain" }, status: 400 }
        );
      }
    });
  }

  storeGoogleTokens(tokens: { access_token: string; refresh_token?: string; expires_in: number }) {
    this.setState({
      ...this.state,
      googleTokens: {
        accessToken: tokens.access_token,
        refreshToken: tokens.refresh_token ?? this.state.googleTokens?.refreshToken,
        expiresAt: Date.now() + tokens.expires_in * 1000,
      },
    });
  }

  @callable()
  async getGoogleTokenStatus() {
    const tokens = this.state.googleTokens;
    if (!tokens) return { connected: false };
    return {
      connected: true,
      hasRefreshToken: !!tokens.refreshToken,
      expiresAt: new Date(tokens.expiresAt).toISOString(),
      isExpired: Date.now() > tokens.expiresAt,
    };
  }

  async disconnectGoogle() {
    const tokens = this.state.googleTokens;

    // Best-effort revoke with Google — doesn't block clearing local state if it fails
    if (tokens?.refreshToken) {
      try {
        await fetch("https://oauth2.googleapis.com/revoke", {
          method: "POST",
          headers: { "Content-Type": "application/x-www-form-urlencoded" },
          body: new URLSearchParams({ token: tokens.refreshToken }),
        });
      } catch (e) {
        console.error("Failed to revoke Google token:", e);
      }
    }

    this.setState({ ...this.state, googleTokens: undefined });
}

  async getValidAccessToken(): Promise<string> {
  const tokens = this.state.googleTokens;
  if (!tokens) throw new Error("Not connected to Google. Please authorize first.");

  const isExpired = Date.now() > tokens.expiresAt - 60_000; // refresh 1 min early
  if (!isExpired) return tokens.accessToken;

  if (!tokens.refreshToken) {
    throw new Error("Access token expired and no refresh token available. Please reauthorize.");
  }

  const res = await fetch("https://oauth2.googleapis.com/token", {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams({
      client_id: this.env.GOOGLE_CLIENT_ID,
      client_secret: this.env.GOOGLE_CLIENT_SECRET,
      refresh_token: tokens.refreshToken,
      grant_type: "refresh_token",
    }),
  });

  if (!res.ok) {
    throw new Error(`Token refresh failed: ${res.status}`);
  }

  const data = (await res.json()) as { access_token: string; expires_in: number };
  this.storeGoogleTokens({
    access_token: data.access_token,
    refresh_token: tokens.refreshToken, // refresh responses don't resend it
    expires_in: data.expires_in,
  });

  return data.access_token;
}

  @callable()
  async addServer(name: string, url: string) {
    return await this.addMcpServer(name, url);
  }

  @callable()
  async removeServer(serverId: string) {
    await this.removeMcpServer(serverId);
  }

  async onChatMessage(_onFinish: unknown, options?: OnChatMessageOptions) {
    const mcpTools = this.mcp.getAITools();
    const workersai = createWorkersAI({ binding: this.env.AI });

    const result = streamText({
      model: workersai("@cf/google/gemma-4-26b-a4b-it", {
        sessionAffinity: this.sessionAffinity
      }),
      system: `You are a helpful assistant that can understand images. You can check the weather, get the user's timezone, run calculations, and schedule tasks. When users share images, describe what you see and answer questions about them.

${getSchedulePrompt({ date: new Date() })}

If the user asks to schedule a task, use the schedule tool to schedule the task.`,
      // Prune old tool calls and reasoning to save tokens on long conversations
      messages: pruneMessages({
        messages: await convertToModelMessages(this.messages),
        toolCalls: "before-last-2-messages",
        reasoning: "before-last-message"
      }),
      tools: {
        // MCP tools from connected servers
        ...mcpTools,

        // Server-side tool: runs automatically on the server
        getWeather: tool({
          description: "Get the current weather for a city",
          inputSchema: z.object({
            city: z.string().describe("City name")
          }),
          execute: async ({ city }) => {
            // Replace with a real weather API in production
            const conditions = ["sunny", "cloudy", "rainy", "snowy"];
            const temp = Math.floor(Math.random() * 30) + 5;
            return {
              city,
              temperature: temp,
              condition:
                conditions[Math.floor(Math.random() * conditions.length)],
              unit: "celsius"
            };
          }
        }),

        // Client-side tool: no execute function — the browser handles it
        getUserTimezone: tool({
          description:
            "Get the user's timezone from their browser. Use this when you need to know the user's local time.",
          inputSchema: z.object({})
        }),

        // Approval tool: requires user confirmation before executing
        calculate: tool({
          description:
            "Perform a math calculation with two numbers. Requires user approval for large numbers.",
          inputSchema: z.object({
            a: z.number().describe("First number"),
            b: z.number().describe("Second number"),
            operator: z
              .enum(["+", "-", "*", "/", "%"])
              .describe("Arithmetic operator")
          }),
          needsApproval: async ({ a, b }) =>
            Math.abs(a) > 1000 || Math.abs(b) > 1000,
          execute: async ({ a, b, operator }) => {
            const ops: Record<string, (x: number, y: number) => number> = {
              "+": (x, y) => x + y,
              "-": (x, y) => x - y,
              "*": (x, y) => x * y,
              "/": (x, y) => x / y,
              "%": (x, y) => x % y
            };
            if (operator === "/" && b === 0) {
              return { error: "Division by zero" };
            }
            return {
              expression: `${a} ${operator} ${b}`,
              result: ops[operator](a, b)
            };
          }
        }),

        scheduleTask: tool({
          description:
            "Schedule a task to be executed at a later time. Use this when the user asks to be reminded or wants something done later.",
          inputSchema: scheduleSchema,
          execute: async ({ when, description }) => {
            if (when.type === "no-schedule") {
              return "Not a valid schedule input";
            }
            const input =
              when.type === "scheduled"
                ? when.date
                : when.type === "delayed"
                  ? when.delayInSeconds
                  : when.type === "cron"
                    ? when.cron
                    : null;
            if (!input) return "Invalid schedule type";
            try {
              this.schedule(input, "executeTask", description, {
                idempotent: true
              });
              return `Task scheduled: "${description}" (${when.type}: ${input})`;
            } catch (error) {
              return `Error scheduling task: ${error}`;
            }
          }
        }),

        getScheduledTasks: tool({
          description: "List all tasks that have been scheduled",
          inputSchema: z.object({}),
          execute: async () => {
            const tasks = this.getSchedules();
            return tasks.length > 0 ? tasks : "No scheduled tasks found.";
          }
        }),

        cancelScheduledTask: tool({
          description: "Cancel a scheduled task by its ID",
          inputSchema: z.object({
            taskId: z.string().describe("The ID of the task to cancel")
          }),
          execute: async ({ taskId }) => {
            try {
              this.cancelSchedule(taskId);
              return `Task ${taskId} cancelled.`;
            } catch (error) {
              return `Error cancelling task: ${error}`;
            }
          }
        }),

        searchJobs: tool({
          description: "Search for job listings by keywords and location using Jooble",
          inputSchema: z.object({
            keywords: z.string().describe("Job title or keywords, e.g. 'data engineer'"),
            location: z.string().optional().describe("City or region to search in"),
            page: z.number().optional().describe("Results page number, defaults to 1"),
          }),
          execute: async ({ keywords, location, page }) => {
            const res = await fetch(`https://jooble.org/api/${this.env.JOOBLE_API_KEY}`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ keywords, location, page }),
            });

            if (!res.ok) {
              return { error: `Jooble API error: ${res.status}` };
            }

            const data = (await res.json()) as JoobleResponse;
            // Trim down to the essentials so the model isn't flooded with raw HTML/snippets
            return {
              totalCount: data.totalCount,
              jobs: (data.jobs ?? []).slice(0, 10).map((job: any) => ({
                title: job.title,
                company: job.company,
                location: job.location,
                salary: job.salary,
                link: job.link,
              })),
            };
          },
        }),

        listInbox: tool({
          description: "List recent emails from the user's Gmail inbox",
          inputSchema: z.object({
            maxResults: z.number().optional().describe("Number of emails to fetch, defaults to 10"),
          }),
          execute: async ({ maxResults = 10 }) => {
            const accessToken = await this.getValidAccessToken();

            const listRes = await fetch(
              `https://gmail.googleapis.com/gmail/v1/users/me/messages?maxResults=${maxResults}`,
              { headers: { Authorization: `Bearer ${accessToken}` } }
            );
            if (!listRes.ok) return { error: `Gmail list error: ${listRes.status}` };

            const listData = (await listRes.json()) as { messages?: { id: string }[] };
            if (!listData.messages) return { emails: [] };

            // Fetch metadata (subject/from) for each message
            const emails = await Promise.all(
              listData.messages.map(async (msg) => {
                const msgRes = await fetch(
                  `https://gmail.googleapis.com/gmail/v1/users/me/messages/${msg.id}?format=metadata&metadataHeaders=Subject&metadataHeaders=From`,
                  { headers: { Authorization: `Bearer ${accessToken}` } }
                );
                const msgData = (await msgRes.json()) as {
                  id: string;
                  snippet: string;
                  payload: { headers: { name: string; value: string }[] };
                };
                const headers = msgData.payload?.headers ?? [];
                return {
                  id: msgData.id,
                  subject: headers.find((h) => h.name === "Subject")?.value ?? "(no subject)",
                  from: headers.find((h) => h.name === "From")?.value ?? "(unknown)",
                  snippet: msgData.snippet,
                };
              })
            );

            return { emails };
          },
        }),

        sendEmail: tool({
          description: "Send an email from the user's Gmail account",
          inputSchema: z.object({
            to: z.string().describe("Recipient email address"),
            subject: z.string().describe("Email subject line"),
            body: z.string().describe("Email body text"),
          }),
          needsApproval: async () => true, // always confirm before sending — irreversible action
          execute: async ({ to, subject, body }) => {
            const accessToken = await this.getValidAccessToken();

            const message = [`To: ${to}`, `Subject: ${subject}`, "Content-Type: text/plain; charset=utf-8", "", body].join("\n");
            const encoded = btoa(message).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");

            const res = await fetch("https://gmail.googleapis.com/gmail/v1/users/me/messages/send", {
              method: "POST",
              headers: {
                Authorization: `Bearer ${accessToken}`,
                "Content-Type": "application/json",
              },
              body: JSON.stringify({ raw: encoded }),
            });

            if (!res.ok) {
              const errText = await res.text();
              return { error: `Gmail send error: ${res.status} — ${errText}` };
            }

            return { success: true, to, subject };
          },
        }),

        connectGmail: tool({
          description: "Provide the user a link to authorize Gmail access when they want to connect their Google account or if a Gmail tool fails due to missing authorization",
          inputSchema: z.object({}),
          execute: async () => {
            return {
              message: "Click this link to connect your Gmail account",
              url: `${this.env.APP_URL}/oauth/login?agent=${this.name}`,
            };
          },
        }),

        disconnectGmail: tool({
          description: "Disconnect the user's Gmail account, revoking access and forgetting stored credentials",
          inputSchema: z.object({}),
          needsApproval: async () => true, // confirm before revoking access
          execute: async () => {
            await this.disconnectGoogle();
            return { message: "Gmail account disconnected." };
          },
        }),

      },
      stopWhen: stepCountIs(20),
      abortSignal: options?.abortSignal
    });

    return result.toUIMessageStreamResponse();
  }

  async executeTask(description: string, _task: Schedule<string>) {
    // Do the actual work here (send email, call API, etc.)
    console.log(`Executing scheduled task: ${description}`);

    // Notify connected clients via a broadcast event.
    // We use broadcast() instead of saveMessages() to avoid injecting
    // into chat history — that would cause the AI to see the notification
    // as new context and potentially loop.
    this.broadcast(
      JSON.stringify({
        type: "scheduled-task",
        description,
        timestamp: new Date().toISOString()
      })
    );
  }
}

function getGoogleAuthUrl(env: Env, agentName: string) {
  const params = new URLSearchParams({
    client_id: env.GOOGLE_CLIENT_ID,
    redirect_uri: env.GOOGLE_REDIRECT_URI,
    response_type: "code",
    scope: "https://www.googleapis.com/auth/gmail.readonly https://www.googleapis.com/auth/gmail.send",
    access_type: "offline",
    prompt: "consent",
    state: agentName,
  });
  return `https://accounts.google.com/o/oauth2/v2/auth?${params}`;
}

async function handleOAuthCallback(request: Request, env: Env) {
  const url = new URL(request.url);
  const code = url.searchParams.get("code");
  const agentName = url.searchParams.get("state");

  if (!code || !agentName) {
    return new Response("Missing code or state", { status: 400 });
  }

  const tokenRes = await fetch("https://oauth2.googleapis.com/token", {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams({
      code,
      client_id: env.GOOGLE_CLIENT_ID,
      client_secret: env.GOOGLE_CLIENT_SECRET,
      redirect_uri: env.GOOGLE_REDIRECT_URI,
      grant_type: "authorization_code",
    }),
  });

  const tokens = await tokenRes.json() as {
    access_token: string;
    refresh_token?: string;
    expires_in: number;
  };

  const agent = await getAgentByName(env.ChatAgent, agentName);
  await agent.storeGoogleTokens(tokens);

  return new Response("Authorized! You can close this tab.");
}

export default {
  async fetch(request: Request, env: Env) {

    const url = new URL(request.url);

    if (url.pathname === "/oauth/login") {
      const agentName = url.searchParams.get("agent");
      if (!agentName) return new Response("Missing agent name", { status: 400 });
      return Response.redirect(getGoogleAuthUrl(env, agentName), 302);
    }

    if (url.pathname === "/oauth/callback") {
      return handleOAuthCallback(request, env);
    }

    return (
      (await routeAgentRequest(request, env)) ||
      new Response("Not found", { status: 404 })
    );
  }
} satisfies ExportedHandler<Env>;
