import React, { useState, useEffect, useRef } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { User, Contact, UserRole } from './types';
import { StorageService } from './services/storageService';
import { ExcelService } from './services/excelService';
import { ContactForm } from './components/ContactForm';
import { 
  LogOut, 
  User as UserIcon, 
  Search, 
  Star, 
  Plus, 
  Upload, 
  Download, 
  Phone, 
  Mail, 
  MapPin, 
  Trash2, 
  Edit2,
  Shield,
  Briefcase
} from 'lucide-react';

// --- Login Component ---
const Login: React.FC<{ onLogin: (u: User) => void }> = ({ onLogin }) => {
  const [mode, setMode] = useState<'USER' | 'ADMIN'>('USER');
  const [isRegister, setIsRegister] = useState(false);
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');

  useEffect(() => {
    setError('');
    setIsRegister(false);
    setUsername('');
    setPassword('');
  }, [mode]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setError('');

    if (mode === 'USER' && isRegister) {
      if (StorageService.findUser(username)) {
        setError('Username already taken');
        return;
      }
      const newUser: User = {
        id: uuidv4(),
        username,
        password,
        role: UserRole.USER,
        createdAt: Date.now()
      };
      StorageService.addUser(newUser);
      onLogin(newUser);
    } else {
      const user = StorageService.findUser(username);
      if (user && user.password === password) {
        if (mode === 'ADMIN' && user.role !== UserRole.ADMIN) {
          setError('Access denied: Not an administrator.');
          return;
        }
        onLogin(user);
      } else {
        setError('Invalid username or password.');
      }
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-500 to-indigo-600 p-4">
      <div className="bg-white p-8 rounded-2xl shadow-2xl w-full max-w-md">
        <h1 className="text-3xl font-bold text-center mb-6 text-slate-800">Smart Connect</h1>
        
        <div className="flex p-1 bg-slate-100 rounded-lg mb-6">
          <button 
            onClick={() => setMode('USER')}
            className={`flex-1 py-2 text-sm font-medium rounded-md transition flex items-center justify-center gap-2 ${mode === 'USER' ? 'bg-white shadow text-blue-600' : 'text-slate-500 hover:text-slate-700'}`}
          >
            <UserIcon size={16} /> User
          </button>
          <button 
            onClick={() => setMode('ADMIN')}
            className={`flex-1 py-2 text-sm font-medium rounded-md transition flex items-center justify-center gap-2 ${mode === 'ADMIN' ? 'bg-white shadow text-purple-600' : 'text-slate-500 hover:text-slate-700'}`}
          >
            <Shield size={16} /> Admin
          </button>
        </div>

        <h2 className="text-xl font-semibold text-center mb-6 text-slate-700">
          {mode === 'ADMIN' ? 'Administrator Access' : (isRegister ? 'Create Account' : 'User Sign In')}
        </h2>

        {error && (
          <div className="mb-4 p-3 bg-red-50 text-red-600 text-sm rounded-lg border border-red-100">
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">Username</label>
            <input
              type="text"
              required
              className="w-full p-2.5 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition"
              value={username}
              onChange={e => setUsername(e.target.value)}
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">Password</label>
            <input
              type="password"
              required
              className="w-full p-2.5 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition"
              value={password}
              onChange={e => setPassword(e.target.value)}
            />
          </div>
          <button
            type="submit"
            className={`w-full py-3 text-white rounded-lg font-semibold shadow-lg hover:shadow-xl transition transform hover:-translate-y-0.5 ${mode === 'ADMIN' ? 'bg-purple-600 hover:bg-purple-700' : 'bg-blue-600 hover:bg-blue-700'}`}
          >
            {mode === 'ADMIN' ? 'Login as Admin' : (isRegister ? 'Sign Up' : 'Sign In')}
          </button>
        </form>

        {mode === 'USER' && (
          <div className="mt-6 text-center">
            <button 
              onClick={() => setIsRegister(!isRegister)}
              className="text-sm text-blue-600 hover:text-blue-800 font-medium"
            >
              {isRegister ? 'Already have an account? Sign In' : 'Need an account? Create one'}
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

// --- Admin Dashboard ---
const AdminDashboard: React.FC<{ currentUser: User; onLogout: () => void }> = ({ currentUser, onLogout }) => {
  const [users, setUsers] = useState<User[]>([]);

  useEffect(() => {
    setUsers(StorageService.getUsers());
  }, []);

  const handleDeleteUser = (userId: string) => {
    if (window.confirm('Are you sure? This will delete the user and all their contacts permanently.')) {
      StorageService.deleteUser(userId);
      setUsers(StorageService.getUsers());
    }
  };

  return (
    <div className="min-h-screen bg-slate-50">
      <header className="bg-purple-700 text-white p-4 shadow-lg sticky top-0 z-10">
        <div className="max-w-7xl mx-auto flex justify-between items-center">
          <div className="flex items-center gap-2">
            <Shield size={24} />
            <h1 className="text-xl font-bold">Admin Console</h1>
          </div>
          <div className="flex items-center gap-4">
            <span className="text-purple-200">Logged in as {currentUser.username}</span>
            <button onClick={onLogout} className="p-2 hover:bg-purple-600 rounded-full transition" title="Logout">
              <LogOut size={20} />
            </button>
          </div>
        </div>
      </header>
      
      <main className="max-w-7xl mx-auto p-6">
        <div className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
          <div className="p-6 border-b border-slate-100">
            <h2 className="text-lg font-semibold text-slate-800">System Users</h2>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-left">
              <thead className="bg-slate-50 text-slate-500 font-medium border-b border-slate-100">
                <tr>
                  <th className="p-4">Username</th>
                  <th className="p-4">Role</th>
                  <th className="p-4">Created At</th>
                  <th className="p-4 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {users.map(u => (
                  <tr key={u.id} className="hover:bg-slate-50 transition">
                    <td className="p-4 font-medium text-slate-700">{u.username}</td>
                    <td className="p-4">
                      <span className={`px-2 py-1 rounded-full text-xs font-semibold ${u.role === 'ADMIN' ? 'bg-purple-100 text-purple-700' : 'bg-blue-100 text-blue-700'}`}>
                        {u.role}
                      </span>
                    </td>
                    <td className="p-4 text-slate-500 text-sm">
                      {new Date(u.createdAt).toLocaleDateString()}
                    </td>
                    <td className="p-4 text-right">
                      {u.id !== currentUser.id && (
                        <button 
                          onClick={() => handleDeleteUser(u.id)}
                          className="text-red-500 hover:bg-red-50 p-2 rounded-md transition flex items-center justify-center ml-auto"
                          title="Delete User"
                        >
                          <Trash2 size={18} />
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </main>
    </div>
  );
};

// --- User Dashboard ---
const UserDashboard: React.FC<{ currentUser: User; onLogout: () => void }> = ({ currentUser, onLogout }) => {
  const [contacts, setContacts] = useState<Contact[]>([]);
  const [search, setSearch] = useState('');
  const [showFavorites, setShowFavorites] = useState(false);
  const [isFormOpen, setIsFormOpen] = useState(false);
  const [editingContact, setEditingContact] = useState<Contact | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const refreshContacts = () => {
    setContacts(StorageService.getContactsByUser(currentUser.id));
  };

  useEffect(() => {
    refreshContacts();
  }, [currentUser.id]);

  const filteredContacts = contacts.filter(c => {
    const matchesSearch = c.fullName.toLowerCase().includes(search.toLowerCase()) ||
                          c.company?.toLowerCase().includes(search.toLowerCase());
    const matchesFav = showFavorites ? c.isFavorite : true;
    return matchesSearch && matchesFav;
  });

  const handleToggleFavorite = (id: string) => {
    StorageService.toggleFavorite(id);
    refreshContacts();
  };

  const handleDelete = (id: string) => {
    if (window.confirm('Delete this contact?')) {
      StorageService.deleteContact(id);
      refreshContacts();
    }
  };

  const handleExport = () => {
    ExcelService.exportContacts(contacts);
  };

  const handleImportClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      try {
        const newContacts = await ExcelService.importContacts(e.target.files[0], currentUser.id);
        newContacts.forEach(c => StorageService.saveContact(c));
        refreshContacts();
        alert(`Successfully imported ${newContacts.length} contacts.`);
      } catch (err) {
        alert('Error importing file. Please ensure valid Excel format.');
        console.error(err);
      }
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const handleSaveContact = (contact: Contact) => {
    StorageService.saveContact(contact);
    refreshContacts();
    setIsFormOpen(false);
    setEditingContact(null);
  };

  return (
    <div className="min-h-screen bg-slate-50 flex flex-col">
      <header className="bg-white shadow-sm sticky top-0 z-20">
        <div className="max-w-7xl mx-auto px-4 py-3 flex justify-between items-center">
          <div className="flex items-center gap-2 text-blue-600">
            <Briefcase size={28} className="text-blue-600"/>
            <h1 className="text-xl font-bold text-slate-800">Smart Connect</h1>
          </div>
          <div className="flex items-center gap-4">
             <div className="hidden md:flex items-center gap-2 text-sm text-slate-600 bg-slate-100 px-3 py-1.5 rounded-full">
                <UserIcon size={14} />
                <span className="font-medium">{currentUser.username}</span>
             </div>
            <button onClick={onLogout} className="text-slate-500 hover:text-red-600 transition" title="Logout">
              <LogOut size={20} />
            </button>
          </div>
        </div>
      </header>

      <div className="flex-1 max-w-7xl mx-auto w-full p-4 md:p-6 space-y-6">
        
        {/* Toolbar */}
        <div className="flex flex-col md:flex-row gap-4 justify-between items-center bg-white p-4 rounded-xl shadow-sm border border-slate-200">
            <div className="flex items-center gap-4 w-full md:w-auto flex-1">
                <div className="relative w-full md:w-80">
                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" size={18} />
                    <input 
                        type="text" 
                        placeholder="Search contacts..." 
                        className="w-full pl-10 pr-4 py-2 bg-slate-50 border border-slate-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:outline-none transition"
                        value={search}
                        onChange={e => setSearch(e.target.value)}
                    />
                </div>
                
                <div className="hidden md:flex bg-slate-100 p-1 rounded-lg shrink-0">
                    <button
                        onClick={() => setShowFavorites(false)}
                        className={`px-3 py-1.5 text-sm font-medium rounded-md transition ${!showFavorites ? 'bg-white text-slate-800 shadow-sm' : 'text-slate-500 hover:text-slate-700'}`}
                    >
                        All
                    </button>
                    <button
                        onClick={() => setShowFavorites(true)}
                        className={`px-3 py-1.5 text-sm font-medium rounded-md transition flex items-center gap-1.5 ${showFavorites ? 'bg-white text-yellow-600 shadow-sm' : 'text-slate-500 hover:text-slate-700'}`}
                    >
                        <Star size={14} fill={showFavorites ? "currentColor" : "none"} /> Favorites
                    </button>
                </div>
            </div>

            <div className="flex gap-2 w-full md:w-auto justify-end items-center">
                 {/* Mobile Tabs */}
                 <div className="md:hidden flex bg-slate-100 p-1 rounded-lg mr-auto">
                    <button
                        onClick={() => setShowFavorites(false)}
                        className={`px-3 py-1.5 text-sm font-medium rounded-md transition ${!showFavorites ? 'bg-white text-slate-800 shadow-sm' : 'text-slate-500 hover:text-slate-700'}`}
                    >
                        All
                    </button>
                    <button
                        onClick={() => setShowFavorites(true)}
                        className={`px-3 py-1.5 text-sm font-medium rounded-md transition flex items-center gap-1.5 ${showFavorites ? 'bg-white text-yellow-600 shadow-sm' : 'text-slate-500 hover:text-slate-700'}`}
                    >
                        <Star size={14} fill={showFavorites ? "currentColor" : "none"} />
                    </button>
                 </div>

                 <div className="h-9 w-px bg-slate-200 mx-1 hidden md:block"></div>

                 <button onClick={handleExport} className="p-2 text-slate-600 hover:bg-slate-100 rounded-lg transition" title="Export Excel">
                    <Download size={20} />
                 </button>
                 <button onClick={handleImportClick} className="p-2 text-slate-600 hover:bg-slate-100 rounded-lg transition" title="Import Excel">
                    <Upload size={20} />
                 </button>
                 <input type="file" ref={fileInputRef} className="hidden" accept=".xlsx, .xls" onChange={handleFileChange} />

                 <button 
                    onClick={() => { setEditingContact(null); setIsFormOpen(true); }}
                    className="ml-2 flex items-center gap-2 bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition shadow-sm font-medium"
                 >
                    <Plus size={18} />
                    <span className="hidden sm:inline">Add Contact</span>
                 </button>
            </div>
        </div>

        {/* Contact Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredContacts.map(contact => (
                <div key={contact.id} className="bg-white rounded-xl shadow-sm border border-slate-200 hover:shadow-md transition group overflow-hidden flex flex-col">
                    <div className="p-5 flex-1">
                        <div className="flex justify-between items-start mb-3">
                            <div>
                                <h3 className="text-lg font-bold text-slate-800 line-clamp-1">{contact.fullName}</h3>
                                {(contact.jobTitle || contact.company) && (
                                    <p className="text-sm text-slate-500 line-clamp-1">
                                        {[contact.jobTitle, contact.company].filter(Boolean).join(' • ')}
                                    </p>
                                )}
                            </div>
                            <button 
                                onClick={(e) => { e.stopPropagation(); handleToggleFavorite(contact.id); }}
                                className={`text-slate-300 hover:text-yellow-400 transition ${contact.isFavorite ? 'text-yellow-400' : ''}`}
                            >
                                <Star size={20} fill={contact.isFavorite ? "currentColor" : "none"} />
                            </button>
                        </div>

                        <div className="space-y-2 mt-4">
                            {contact.phones.slice(0, 2).map(phone => (
                                <div key={phone.id} className="flex items-center gap-3 text-sm text-slate-600">
                                    <Phone size={14} className="text-slate-400 shrink-0" />
                                    <span className="truncate">{phone.value}</span>
                                    <span className="text-xs bg-slate-100 px-1.5 py-0.5 rounded text-slate-500">{phone.label}</span>
                                </div>
                            ))}
                            {contact.emails.slice(0, 2).map(email => (
                                <div key={email.id} className="flex items-center gap-3 text-sm text-slate-600">
                                    <Mail size={14} className="text-slate-400 shrink-0" />
                                    <span className="truncate">{email.value}</span>
                                    <span className="text-xs bg-slate-100 px-1.5 py-0.5 rounded text-slate-500">{email.label}</span>
                                </div>
                            ))}
                             {contact.addresses.slice(0, 1).map(addr => (
                                <div key={addr.id} className="flex items-center gap-3 text-sm text-slate-600">
                                    <MapPin size={14} className="text-slate-400 shrink-0" />
                                    <span className="truncate">{addr.city}, {addr.street}</span>
                                </div>
                            ))}
                        </div>
                    </div>
                    
                    <div className="bg-slate-50 p-3 flex justify-end gap-2 border-t border-slate-100">
                         <button 
                            onClick={() => { setEditingContact(contact); setIsFormOpen(true); }}
                            className="p-1.5 text-slate-500 hover:text-blue-600 hover:bg-blue-50 rounded-md transition"
                            title="Edit"
                         >
                            <Edit2 size={16} />
                         </button>
                         <button 
                            onClick={() => handleDelete(contact.id)}
                            className="p-1.5 text-slate-500 hover:text-red-600 hover:bg-red-50 rounded-md transition"
                            title="Delete"
                         >
                            <Trash2 size={16} />
                         </button>
                    </div>
                </div>
            ))}
            
            {filteredContacts.length === 0 && (
                <div className="col-span-full flex flex-col items-center justify-center py-20 text-slate-400 bg-white rounded-xl border border-dashed border-slate-300">
                    {showFavorites ? (
                        <>
                            <Star size={48} className="mb-4 opacity-50 text-yellow-400" />
                            <p className="text-lg font-medium">No favorite contacts</p>
                            <p className="text-sm">Mark contacts with a star to see them here.</p>
                        </>
                    ) : (
                        <>
                            <UserIcon size={48} className="mb-4 opacity-50" />
                            <p className="text-lg font-medium">No contacts found</p>
                            <p className="text-sm">Add a new contact or try a different search.</p>
                        </>
                    )}
                </div>
            )}
        </div>
      </div>

      {isFormOpen && (
        <ContactForm 
            userId={currentUser.id}
            existingContact={editingContact}
            onClose={() => { setIsFormOpen(false); setEditingContact(null); }}
            onSave={handleSaveContact}
        />
      )}
    </div>
  );
};

// --- Main App ---
const App: React.FC = () => {
  const [user, setUser] = useState<User | null>(null);

  useEffect(() => {
    const session = localStorage.getItem('app_session');
    if (session) {
      setUser(JSON.parse(session));
    }
  }, []);

  const handleLogin = (u: User) => {
    setUser(u);
    localStorage.setItem('app_session', JSON.stringify(u));
  };

  const handleLogout = () => {
    setUser(null);
    localStorage.removeItem('app_session');
  };

  if (!user) {
    return <Login onLogin={handleLogin} />;
  }

  return user.role === UserRole.ADMIN 
    ? <AdminDashboard currentUser={user} onLogout={handleLogout} /> 
    : <UserDashboard currentUser={user} onLogout={handleLogout} />;
};

export default App;
