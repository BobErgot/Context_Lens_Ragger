"use client";

import React, { useEffect, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";
import { RemoteRunnable } from "@langchain/core/runnables/remote";
import { applyPatch } from "@langchain/core/utils/json_patch";

import { EmptyState } from "./EmptyState";
import { ChatMessageBubble, Message } from "./ChatMessageBubble";
import { AutoResizeTextarea } from "./AutoResizeTextarea";
import { marked } from "marked";
import { Renderer } from "marked";
import hljs from "highlight.js";
import "highlight.js/styles/gradient-dark.css";
import {
  useToast,
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalCloseButton,
  Button,
  List,
  ListItem,
  IconButton as ChakraIconButton, ModalFooter
} from "@chakra-ui/react";
import "react-toastify/dist/ReactToastify.css";
import {
  Heading,
  Flex,
  IconButton,
  InputGroup,
  InputRightElement,
  Spinner,
  Select,
  Link
} from "@chakra-ui/react";
import { ArrowUpIcon, AttachmentIcon, DeleteIcon } from "@chakra-ui/icons";
import { Source } from "./SourceBubble";
import { apiBaseUrl } from "../utils/constants";

const MODEL_TYPES = [
  "openai_gpt_3_5_turbo",
  "anthropic_claude_3_haiku",
  "google_gemini_pro",
  "fireworks_mixtral",
  "cohere_command",
];

const defaultLlmValue =
    MODEL_TYPES[Math.floor(Math.random() * MODEL_TYPES.length)];

export function ChatWindow(props: { conversationId: string }) {
  const conversationId = props.conversationId;

  const searchParams = useSearchParams();

  const messageContainerRef = useRef<HTMLDivElement | null>(null);
  const [messages, setMessages] = useState<Array<Message>>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const toast = useToast();
  const [llm, setLlm] = useState(
      searchParams.get("llm") ?? "openai_gpt_3_5_turbo",
  );
  const [llmIsLoading, setLlmIsLoading] = useState(true);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [uploadedDocuments, setUploadedDocuments] = useState<string[]>([]);
  useEffect(() => {
    setLlm(searchParams.get("llm") ?? defaultLlmValue);
    setLlmIsLoading(false);
    fetchUploadedDocuments();
  }, []);

  const fetchUploadedDocuments = async () => {
    try {
      const response = await fetch(`${apiBaseUrl}/documents/`);
      const data = await response.json();
      setUploadedDocuments(data.documents);
    } catch (error) {
      console.error("Failed to fetch documents", error);
    }
  };

  const [chatHistory, setChatHistory] = useState<
      { human: string; ai: string }[]
  >([]);

  const sendMessage = async (message?: string) => {
    if (messageContainerRef.current) {
      messageContainerRef.current.classList.add("grow");
    }
    if (isLoading) {
      return;
    }
    const messageValue = message ?? input;
    if (messageValue === "") return;
    setInput("");
    setMessages((prevMessages) => [
      ...prevMessages,
      { id: Math.random().toString(), content: messageValue, role: "user" },
    ]);
    setIsLoading(true);

    let accumulatedMessage = "";
    let runId: string | undefined = undefined;
    let sources: Source[] | undefined = undefined;
    let messageIndex: number | null = null;

    let renderer = new Renderer();
    renderer.paragraph = (text) => {
      return text + "\n";
    };
    renderer.list = (text) => {
      return `${text}\n\n`;
    };
    renderer.listitem = (text) => {
      return `\n• ${text}`;
    };
    renderer.code = (code, language) => {
      const validLanguage = hljs.getLanguage(language || "")
          ? language
          : "plaintext";
      const highlightedCode = hljs.highlight(
          validLanguage || "plaintext",
          code,
      ).value;
      return `<pre class="highlight bg-gray-700" style="padding: 5px; border-radius: 5px; overflow: auto; overflow-wrap: anywhere; white-space: pre-wrap; max-width: 100%; display: block; line-height: 1.2"><code class="${language}" style="color: #d6e2ef; font-size: 12px; ">${highlightedCode}</code></pre>`;
    };
    marked.setOptions({ renderer });
    try {
      const sourceStepName = "FindDocs";
      let streamedResponse: Record<string, any> = {};
      const remoteChain = new RemoteRunnable({
        url: apiBaseUrl + "/chat",
        options: {
          timeout: 60000,
        },
      });
      const llmDisplayName = llm ?? "openai_gpt_3_5_turbo";
      const streamLog = await remoteChain.streamLog(
          {
            question: messageValue,
            chat_history: chatHistory,
          },
          {
            configurable: {
              llm: llmDisplayName,
            },
            tags: ["model:" + llmDisplayName],
            metadata: {
              conversation_id: conversationId,
              llm: llmDisplayName,
            },
          },
          {
            includeNames: [sourceStepName],
          },
      );
      for await (const chunk of streamLog) {
        streamedResponse = applyPatch(streamedResponse, chunk.ops, undefined, false).newDocument;
        if (
            Array.isArray(
                streamedResponse?.logs?.[sourceStepName]?.final_output?.output,
            )
        ) {
          sources = streamedResponse.logs[
              sourceStepName
              ].final_output.output.map((doc: Record<string, any>) => ({
            url: doc.metadata.source,
            title: doc.metadata.title,
          }));
        }
        if (streamedResponse.id !== undefined) {
          runId = streamedResponse.id;
        }
        if (Array.isArray(streamedResponse?.streamed_output)) {
          accumulatedMessage = streamedResponse.streamed_output.join("");
        }
        const parsedResult = marked.parse(accumulatedMessage);

        setMessages((prevMessages) => {
          let newMessages = [...prevMessages];
          if (
              messageIndex === null ||
              newMessages[messageIndex] === undefined
          ) {
            messageIndex = newMessages.length;
            newMessages.push({
              id: Math.random().toString(),
              content: parsedResult.trim(),
              runId: runId,
              sources: sources,
              role: "assistant",
            });
          } else if (newMessages[messageIndex] !== undefined) {
            newMessages[messageIndex].content = parsedResult.trim();
            newMessages[messageIndex].runId = runId;
            newMessages[messageIndex].sources = sources;
          }
          return newMessages;
        });
      }
      setChatHistory((prevChatHistory) => [
        ...prevChatHistory,
        { human: messageValue, ai: accumulatedMessage },
      ]);
      setIsLoading(false);
    } catch (e) {
      setMessages((prevMessages) => prevMessages.slice(0, -1));
      setIsLoading(false);
      setInput(messageValue);
      throw e;
    }
  };


  const handleFileSelection = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
    }
  };

  const handleSave = async () => {
    if (!selectedFile) return;

    setIsUploading(true);
    setIsModalOpen(false); // Close the modal when Save is pressed

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch(`${apiBaseUrl}/upload-document/`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Failed to upload document");
      }

      toast({
        title: "Document uploaded successfully.",
        status: "success",
        duration: 3000,
        isClosable: true,
      });

      fetchUploadedDocuments();
    } catch (error) {
      toast({
        title: "Error uploading document.",
        description: (error as Error).message,
        status: "error",
        duration: 3000,
        isClosable: true,
      });
    } finally {
      setIsUploading(false);
      setSelectedFile(null); // Reset the selected file
    }
  };

  const handleDeleteDocument = async (documentId: string) => {
    try {
      const response = await fetch(`${apiBaseUrl}/documents/${documentId}`, {
        method: "DELETE",
      });

      if (!response.ok) {
        throw new Error("Failed to delete document");
      }

      toast({
        title: "Document deleted successfully.",
        status: "success",
        duration: 3000,
        isClosable: true,
      });

      fetchUploadedDocuments();
    } catch (error) {
      toast({
        title: "Error deleting document.",
        description: (error as Error).message,
        status: "error",
        duration: 3000,
        isClosable: true,
      });
    }
  };


  const sendInitialQuestion = async (question: string) => {
    await sendMessage(question);
  };

  const insertUrlParam = (key: string, value?: string) => {
    if (window.history.pushState) {
      const searchParams = new URLSearchParams(window.location.search);
      searchParams.set(key, value ?? "");
      const newurl =
          window.location.protocol +
          "//" +
          window.location.host +
          window.location.pathname +
          "?" +
          searchParams.toString();
      window.history.pushState({ path: newurl }, "", newurl);
    }
  };

  return (
      <div className="flex flex-col items-center p-8 rounded grow max-h-full">
        <Flex
            direction={"column"}
            alignItems={"center"}
            marginTop={messages.length > 0 ? "" : "64px"}
        >
          <Heading
              fontSize={messages.length > 0 ? "2xl" : "3xl"}
              fontWeight={"medium"}
              mb={1}
              color={"#262629"}
          >
            Sia Chat QnA
          </Heading>
          {messages.length > 0 ? (
              <Heading fontSize="md" fontWeight={"normal"} mb={1} color={"#262629"}>
                Let us read the textbook and you can chat. I can assist you in learning.
              </Heading>
          ) : (
              <Heading
                  fontSize="xl"
                  fontWeight={"normal"}
                  color={"#262629"}
                  marginTop={"10px"}
                  textAlign={"center"}
              >
                Let me read the textbook and you can chat. I can assist you in learning.
              </Heading>
          )}
          <div className="text-[#262629] flex flex-wrap items-center mt-4">
            <div className="flex items-center mb-2">
              <span className="shrink-0 mr-2">Powered by</span>
              {llmIsLoading ? (
                  <Spinner className="my-2"></Spinner>
              ) : (
                  <Select
                      value={llm}
                      onChange={(e) => {
                        insertUrlParam("llm", e.target.value);
                        setLlm(e.target.value);
                      }}
                      width={"240px"}
                  >
                    <option value="openai_gpt_3_5_turbo">GPT-3.5-Turbo</option>
                    <option value="anthropic_claude_3_haiku">Claude 3 Haiku</option>
                    <option value="google_gemini_pro">Google Gemini Pro</option>
                    <option value="fireworks_mixtral">
                      Mixtral (via Fireworks.ai)
                    </option>
                    <option value="cohere_command">Cohere</option>
                  </Select>
              )}
            </div>
          </div>
        </Flex>
        <div
            className="flex flex-col-reverse w-full mb-2 overflow-auto"
            ref={messageContainerRef}
        >
          {messages.length > 0 ? (
              [...messages]
                  .reverse()
                  .map((m, index) => (
                      <ChatMessageBubble
                          key={m.id}
                          message={{ ...m }}
                          aiEmoji="𓂀"
                          isMostRecent={index === 0}
                          messageCompleted={!isLoading}
                      ></ChatMessageBubble>
                  ))
          ) : (
              <EmptyState onChoice={sendInitialQuestion} />
          )}
        </div>
        <InputGroup size="md" alignItems={"center"}>
          <AutoResizeTextarea
            value={input}
            maxRows={5}
            marginRight={"56px"}
            placeholder="What does RunnablePassthrough.assign() do?"
            textColor={"#262629"}
            borderColor={"rgb(200, 200, 200)"}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
              } else if (e.key === "Enter" && e.shiftKey) {
                e.preventDefault();
                setInput(input + "\n");
              }
            }}
          />
          <input
            id="fileUpload"
            type="file"
            accept=".pdf,.docx,.jpg,.png,.jpeg,.svg"
            style={{ display: "none" }}
            onChange={handleFileSelection}
          />
          <IconButton
            icon={isUploading ? <Spinner /> : <AttachmentIcon />}
            aria-label="Upload Document"
            colorScheme="blue"
            rounded={"full"}
            marginRight={2}
            onClick={() => setIsModalOpen(true)}
          />
          <IconButton
            colorScheme="blue"
            rounded={"full"}
            aria-label="Send"
            icon={isLoading ? <Spinner /> : <ArrowUpIcon />}
            type="submit"
            onClick={(e) => {
              e.preventDefault();
              sendMessage();
            }}
          />
        </InputGroup>

        <Modal isOpen={isModalOpen} onClose={() => setIsModalOpen(false)}>
          <ModalOverlay />
          <ModalContent>
            <ModalHeader>Upload or Manage Documents</ModalHeader>
            <ModalCloseButton />
            <ModalBody>
              <div className="flex flex-col">
                <input
                  id="fileUpload"
                  type="file"
                  accept=".pdf,.docx,.jpg,.png,.jpeg,.svg"
                  style={{ display: "none" }}
                  onChange={handleFileSelection}
                />
                <Button as="label" htmlFor="fileUpload" mb={4} isLoading={isUploading}>
                  Select New Document
                </Button>
                <List spacing={3}>
                  {uploadedDocuments.map((doc) => (
                    <ListItem
                      key={doc.id}
                      display="flex"
                      justifyContent="space-between"
                      alignItems="center"
                      style={{
                        maxWidth: "100%",
                        overflow: "hidden",
                        textOverflow: "ellipsis",
                        whiteSpace: "nowrap",
                      }}
                    >
                    <span style={{ overflow: "hidden", textOverflow: "ellipsis" }}>
                      {doc.filename}
                    </span>
                      <ChakraIconButton
                        icon={<DeleteIcon />}
                        aria-label="Delete Document"
                        colorScheme="red"
                        onClick={() => handleDeleteDocument(doc.id)}
                      />
                    </ListItem>
                  ))}
                </List>
              </div>
            </ModalBody>
            <ModalFooter>
              {selectedFile && (
                <div
                  style={{
                    maxWidth: "70%",
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    whiteSpace: "nowrap",
                    marginRight: "1rem",
                  }}
                >
                  <strong>Selected File: </strong> {selectedFile.name}
                </div>
              )}
              <Button
                colorScheme="blue"
                onClick={handleSave}
                isLoading={isUploading}
                disabled={!selectedFile}
                style={{ flexShrink: 0 }}
              >
                {selectedFile ? "Upload" : "Save"}
              </Button>
            </ModalFooter>
          </ModalContent>
        </Modal>

        {messages.length === 0 ? (
          <footer className="flex justify-center absolute bottom-8">
            <a
              href="https://github.com/BobErgot/Context_Lens_Ragger"
              target="_blank"
              className="text-[#262629] flex items-center"
              >
                <img src="/images/github-mark.svg" className="h-4 mr-1" />
                <span>View Source</span>
              </a>
            </footer>
        ) : (
            ""
        )}
      </div>
  );
}