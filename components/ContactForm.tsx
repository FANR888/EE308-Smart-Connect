import React, { useState, useEffect } from 'react';
import { Contact, ContactMethod, Address } from '../types';
import { v4 as uuidv4 } from 'uuid';
import { X, Plus, Trash2, Wand2 } from 'lucide-react';
import { GeminiService } from '../services/geminiService';

interface Props {
  userId: string;
  existingContact?: Contact | null;
  onClose: () => void;
  onSave: (contact: Contact) => void;
}

const emptyContact = (userId: string): Contact => ({
  id: '',
  userId,
  fullName: '',
  isFavorite: false,
  phones: [],
  emails: [],
  addresses: [],
  createdAt: Date.now()
});

export const ContactForm: React.FC<Props> = ({ userId, existingContact, onClose, onSave }) => {
  const [formData, setFormData] = useState<Contact>(emptyContact(userId));
  const [smartText, setSmartText] = useState('');
  const [isProcessingAI, setIsProcessingAI] = useState(false);
  const [mode, setMode] = useState<'manual' | 'ai'>('manual');

  useEffect(() => {
    if (existingContact) {
      setFormData(existingContact);
      setMode('manual');
    } else {
      setFormData({ ...emptyContact(userId), id: uuidv4() });
    }
  }, [existingContact, userId]);

  const handleAIParse = async () => {
    if (!smartText.trim()) return;
    setIsProcessingAI(true);
    try {
      const contact = await GeminiService.parseContactFromText(smartText, userId);
      setFormData(contact);
      setMode('manual'); // Switch to review mode
    } catch (e) {
      alert("Failed to parse text. Please check API Key or try again.");
    } finally {
      setIsProcessingAI(false);
    }
  };

  const addField = (field: 'phones' | 'emails' | 'addresses') => {
    setFormData(prev => {
      const newItem = field === 'addresses' 
        ? { id: uuidv4(), label: 'Home', street: '', city: '', zip: '' }
        : { id: uuidv4(), label: field === 'emails' ? 'Email' : 'Mobile', value: '' }; 
      return { ...prev, [field]: [...prev[field], newItem] };
    });
  };

  const removeField = (field: 'phones' | 'emails' | 'addresses', id: string) => {
    setFormData(prev => ({
      ...prev,
      [field]: (prev[field] as any[]).filter((item: any) => item.id !== id)
    }));
  };

  const updateField = (field: 'phones' | 'emails', id: string, key: string, value: string) => {
    setFormData(prev => ({
      ...prev,
      [field]: prev[field].map(item => item.id === id ? { ...item, [key]: value } : item)
    }));
  };

  const updateAddress = (id: string, key: keyof Address, value: string) => {
    setFormData(prev => ({
      ...prev,
      addresses: prev.addresses.map(addr => addr.id === id ? { ...addr, [key]: value } : addr)
    }));
  };

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-xl shadow-2xl w-full max-w-2xl max-h-[90vh] overflow-y-auto">
        <div className="p-6 border-b flex justify-between items-center sticky top-0 bg-white z-10">
          <h2 className="text-xl font-bold text-slate-800">
            {existingContact ? 'Edit Contact' : 'New Contact'}
          </h2>
          <button onClick={onClose} className="p-2 hover:bg-slate-100 rounded-full">
            <X size={20} />
          </button>
        </div>

        <div className="p-6 space-y-6">
          {/* AI Toggle for New Contacts */}
          {!existingContact && (
            <div className="flex gap-4 mb-6">
              <button 
                onClick={() => setMode('manual')}
                className={`flex-1 py-2 rounded-lg font-medium transition ${mode === 'manual' ? 'bg-blue-600 text-white' : 'bg-slate-100 text-slate-600'}`}
              >
                Manual Entry
              </button>
              <button 
                onClick={() => setMode('ai')}
                className={`flex-1 py-2 rounded-lg font-medium transition flex items-center justify-center gap-2 ${mode === 'ai' ? 'bg-purple-600 text-white' : 'bg-slate-100 text-purple-600'}`}
              >
                <Wand2 size={16} /> AI Smart Add
              </button>
            </div>
          )}

          {mode === 'ai' ? (
            <div className="space-y-4">
              <p className="text-sm text-slate-500">Paste an email signature or any text containing contact info.</p>
              <textarea
                className="w-full h-40 p-3 border rounded-lg focus:ring-2 focus:ring-purple-500"
                placeholder="e.g. John Doe, Senior Dev at TechCorp. 555-0123, john@example.com..."
                value={smartText}
                onChange={e => setSmartText(e.target.value)}
              />
              <button
                onClick={handleAIParse}
                disabled={isProcessingAI || !smartText}
                className="w-full py-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:opacity-50 flex justify-center items-center gap-2"
              >
                 {isProcessingAI ? 'Analyzing...' : <><Wand2 size={18}/> Extract Info</>}
              </button>
            </div>
          ) : (
            <div className="space-y-6">
              {/* Basic Info */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="col-span-2">
                  <label className="block text-sm font-medium text-slate-700 mb-1">Full Name</label>
                  <input
                    type="text"
                    required
                    className="w-full p-2 border rounded focus:ring-2 focus:ring-blue-500"
                    value={formData.fullName}
                    onChange={e => setFormData({ ...formData, fullName: e.target.value })}
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-1">Company</label>
                  <input
                    type="text"
                    className="w-full p-2 border rounded focus:ring-2 focus:ring-blue-500"
                    value={formData.company || ''}
                    onChange={e => setFormData({ ...formData, company: e.target.value })}
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-1">Job Title</label>
                  <input
                    type="text"
                    className="w-full p-2 border rounded focus:ring-2 focus:ring-blue-500"
                    value={formData.jobTitle || ''}
                    onChange={e => setFormData({ ...formData, jobTitle: e.target.value })}
                  />
                </div>
              </div>

              {/* Dynamic Phones */}
              <div>
                <div className="flex justify-between items-center mb-2">
                  <label className="text-sm font-medium text-slate-700">Phone Numbers</label>
                  <button type="button" onClick={() => addField('phones')} className="text-blue-600 text-sm flex items-center gap-1 hover:underline"><Plus size={14}/> Add Phone</button>
                </div>
                {formData.phones.map((phone) => (
                  <div key={phone.id} className="flex gap-2 mb-2">
                    <input
                      type="text"
                      className="w-1/3 p-2 border rounded bg-slate-50 text-sm"
                      value={phone.label}
                      onChange={e => updateField('phones', phone.id, 'label', e.target.value)}
                    />
                    <input
                      type="text"
                      className="flex-1 p-2 border rounded"
                      value={phone.value}
                      placeholder="Number"
                      onChange={e => updateField('phones', phone.id, 'value', e.target.value)}
                    />
                    <button onClick={() => removeField('phones', phone.id)} className="text-red-500 p-2 hover:bg-red-50 rounded"><Trash2 size={16}/></button>
                  </div>
                ))}
              </div>

              {/* Dynamic Emails */}
              <div>
                <div className="flex justify-between items-center mb-2">
                  <label className="text-sm font-medium text-slate-700">Emails</label>
                  <button type="button" onClick={() => addField('emails')} className="text-blue-600 text-sm flex items-center gap-1 hover:underline"><Plus size={14}/> Add Email</button>
                </div>
                {formData.emails.map((email) => (
                  <div key={email.id} className="flex gap-2 mb-2">
                    <input
                      type="text"
                      className="w-1/3 p-2 border rounded bg-slate-50 text-sm"
                      value={email.label}
                      onChange={e => updateField('emails', email.id, 'label', e.target.value)}
                    />
                    <input
                      type="email"
                      className="flex-1 p-2 border rounded"
                      value={email.value}
                      placeholder="Email Address"
                      onChange={e => updateField('emails', email.id, 'value', e.target.value)}
                    />
                    <button onClick={() => removeField('emails', email.id)} className="text-red-500 p-2 hover:bg-red-50 rounded"><Trash2 size={16}/></button>
                  </div>
                ))}
              </div>

               {/* Dynamic Addresses */}
               <div>
                <div className="flex justify-between items-center mb-2">
                  <label className="text-sm font-medium text-slate-700">Addresses</label>
                  <button type="button" onClick={() => addField('addresses')} className="text-blue-600 text-sm flex items-center gap-1 hover:underline"><Plus size={14}/> Add Address</button>
                </div>
                {formData.addresses.map((addr) => (
                  <div key={addr.id} className="border p-3 rounded-lg mb-2 relative group">
                    <button onClick={() => removeField('addresses', addr.id)} className="absolute top-2 right-2 text-red-500 opacity-0 group-hover:opacity-100 transition"><Trash2 size={16}/></button>
                    <div className="grid grid-cols-2 gap-2">
                        <input className="col-span-2 p-1 border-b text-sm font-medium" value={addr.label} onChange={e => updateAddress(addr.id, 'label', e.target.value)} placeholder="Label (e.g. Home)" />
                        <input className="col-span-2 p-1 border rounded text-sm" value={addr.street} onChange={e => updateAddress(addr.id, 'street', e.target.value)} placeholder="Street" />
                        <input className="p-1 border rounded text-sm" value={addr.city} onChange={e => updateAddress(addr.id, 'city', e.target.value)} placeholder="City" />
                        <input className="p-1 border rounded text-sm" value={addr.zip} onChange={e => updateAddress(addr.id, 'zip', e.target.value)} placeholder="ZIP" />
                    </div>
                  </div>
                ))}
              </div>

            </div>
          )}
        </div>

        <div className="p-6 border-t bg-slate-50 sticky bottom-0 rounded-b-xl flex justify-end gap-3">
            <button onClick={onClose} className="px-4 py-2 text-slate-600 hover:bg-slate-200 rounded-lg">Cancel</button>
            {mode === 'manual' && (
                <button 
                    onClick={() => onSave(formData)} 
                    disabled={!formData.fullName}
                    className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 font-medium"
                >
                    Save Contact
                </button>
            )}
        </div>
      </div>
    </div>
  );
};