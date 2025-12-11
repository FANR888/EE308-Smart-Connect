export enum UserRole {
  USER = 'USER',
  ADMIN = 'ADMIN'
}

export interface User {
  id: string;
  username: string;
  password: string; // In real app, this would be hashed
  role: UserRole;
  createdAt: number;
}

export interface ContactMethod {
  id: string;
  label: string; // e.g., "Mobile", "Work", "Home"
  value: string;
}

export interface Address {
  id: string;
  label: string;
  street: string;
  city: string;
  zip: string;
}

export interface Contact {
  id: string;
  userId: string; // Owner
  fullName: string;
  company?: string;
  jobTitle?: string;
  isFavorite: boolean;
  phones: ContactMethod[];
  emails: ContactMethod[];
  addresses: Address[];
  notes?: string;
  createdAt: number;
}

export interface AuthState {
  user: User | null;
  isAuthenticated: boolean;
}