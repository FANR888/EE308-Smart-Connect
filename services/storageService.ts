import { User, Contact, UserRole } from '../types';
import { v4 as uuidv4 } from 'uuid';

const KEYS = {
  USERS: 'app_users',
  CONTACTS: 'app_contacts',
  SESSION: 'app_session'
};

// Initialize Admin if not exists
const initStorage = () => {
  const usersRaw = localStorage.getItem(KEYS.USERS);
  if (!usersRaw) {
    const adminUser: User = {
      id: 'admin-001',
      username: '123', // Default admin username
      password: '123', // Default admin password
      role: UserRole.ADMIN,
      createdAt: Date.now()
    };
    localStorage.setItem(KEYS.USERS, JSON.stringify([adminUser]));
  }
  if (!localStorage.getItem(KEYS.CONTACTS)) {
    localStorage.setItem(KEYS.CONTACTS, JSON.stringify([]));
  }
};

initStorage();

export const StorageService = {
  // --- User Management ---
  getUsers: (): User[] => {
    return JSON.parse(localStorage.getItem(KEYS.USERS) || '[]');
  },

  addUser: (user: User): void => {
    const users = StorageService.getUsers();
    users.push(user);
    localStorage.setItem(KEYS.USERS, JSON.stringify(users));
  },

  updateUser: (userId: string, data: Partial<User>): User | null => {
    const users = StorageService.getUsers();
    const index = users.findIndex(u => u.id === userId);
    if (index !== -1) {
      users[index] = { ...users[index], ...data };
      localStorage.setItem(KEYS.USERS, JSON.stringify(users));
      return users[index];
    }
    return null;
  },

  deleteUser: (userId: string): void => {
    let users = StorageService.getUsers();
    users = users.filter(u => u.id !== userId);
    localStorage.setItem(KEYS.USERS, JSON.stringify(users));
    
    // Cascade delete contacts
    let contacts = StorageService.getAllContacts();
    contacts = contacts.filter(c => c.userId !== userId);
    localStorage.setItem(KEYS.CONTACTS, JSON.stringify(contacts));
  },

  findUser: (username: string): User | undefined => {
    const users = StorageService.getUsers();
    return users.find(u => u.username === username);
  },

  // --- Contact Management ---
  getAllContacts: (): Contact[] => {
    return JSON.parse(localStorage.getItem(KEYS.CONTACTS) || '[]');
  },

  getContactsByUser: (userId: string): Contact[] => {
    const contacts = StorageService.getAllContacts();
    return contacts.filter(c => c.userId === userId);
  },

  saveContact: (contact: Contact): void => {
    const contacts = StorageService.getAllContacts();
    const index = contacts.findIndex(c => c.id === contact.id);
    if (index >= 0) {
      contacts[index] = contact;
    } else {
      contacts.push(contact);
    }
    localStorage.setItem(KEYS.CONTACTS, JSON.stringify(contacts));
  },

  deleteContact: (contactId: string): void => {
    let contacts = StorageService.getAllContacts();
    contacts = contacts.filter(c => c.id !== contactId);
    localStorage.setItem(KEYS.CONTACTS, JSON.stringify(contacts));
  },

  toggleFavorite: (contactId: string): void => {
    const contacts = StorageService.getAllContacts();
    const contact = contacts.find(c => c.id === contactId);
    if (contact) {
      contact.isFavorite = !contact.isFavorite;
      localStorage.setItem(KEYS.CONTACTS, JSON.stringify(contacts));
    }
  }
};