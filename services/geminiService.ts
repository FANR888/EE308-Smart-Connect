import { GoogleGenAI, Type } from "@google/genai";
import { Contact, ContactMethod, Address } from "../types";
import { v4 as uuidv4 } from 'uuid';

// WARNING: In a real production app, never expose keys in client code.
// This is for demonstration purposes as requested by the persona using process.env.
const ai = new GoogleGenAI({ apiKey: process.env.API_KEY || '' });

export const GeminiService = {
  parseContactFromText: async (text: string, userId: string): Promise<Contact> => {
    if (!process.env.API_KEY) {
      throw new Error("API Key missing");
    }

    const modelId = "gemini-2.5-flash"; // Using fast model for extraction

    const prompt = `Extract contact information from the following text and return a JSON object.
    Text: "${text}"
    `;

    try {
      const response = await ai.models.generateContent({
        model: modelId,
        contents: prompt,
        config: {
          responseMimeType: "application/json",
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              fullName: { type: Type.STRING },
              company: { type: Type.STRING },
              jobTitle: { type: Type.STRING },
              phones: {
                type: Type.ARRAY,
                items: {
                  type: Type.OBJECT,
                  properties: {
                    label: { type: Type.STRING },
                    value: { type: Type.STRING }
                  }
                }
              },
              emails: {
                type: Type.ARRAY,
                items: {
                  type: Type.OBJECT,
                  properties: {
                    label: { type: Type.STRING },
                    value: { type: Type.STRING }
                  }
                }
              },
              addresses: {
                type: Type.ARRAY,
                items: {
                  type: Type.OBJECT,
                  properties: {
                    label: { type: Type.STRING },
                    street: { type: Type.STRING },
                    city: { type: Type.STRING },
                    zip: { type: Type.STRING }
                  }
                }
              }
            }
          }
        }
      });

      const json = JSON.parse(response.text || '{}');

      // Map to our internal Contact interface with IDs
      const contact: Contact = {
        id: uuidv4(),
        userId,
        fullName: json.fullName || 'No Name Found',
        company: json.company,
        jobTitle: json.jobTitle,
        isFavorite: false,
        createdAt: Date.now(),
        phones: (json.phones || []).map((p: any) => ({ id: uuidv4(), label: p.label || 'Mobile', value: p.value })),
        emails: (json.emails || []).map((e: any) => ({ id: uuidv4(), label: e.label || 'Work', value: e.value })),
        addresses: (json.addresses || []).map((a: any) => ({ 
            id: uuidv4(), 
            label: a.label || 'Home', 
            street: a.street || '', 
            city: a.city || '', 
            zip: a.zip || '' 
        }))
      };

      return contact;
    } catch (error) {
      console.error("Gemini Parse Error", error);
      throw new Error("Failed to parse contact using AI.");
    }
  }
};