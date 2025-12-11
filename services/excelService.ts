// import * as XLSX from "https://cdn.sheetjs.com/xlsx-0.20.1/package/xlsx.mjs";
import * as XLSX from "xlsx";
import { Contact, ContactMethod, Address } from "../types";
import { v4 as uuidv4 } from 'uuid';

export const ExcelService = {
  exportContacts: (contacts: Contact[]) => {
    // Flatten data for Excel
    const rows = contacts.map(c => {
      const row: any = {
        'Full Name': c.fullName,
        'Company': c.company || '',
        'Job Title': c.jobTitle || '',
        'Is Favorite': c.isFavorite ? 'Yes' : 'No',
        'Notes': c.notes || ''
      };

      // Flatten Phones (up to 3 for simplicity in column view)
      c.phones.forEach((p, i) => {
        row[`Phone ${i + 1} Label`] = p.label;
        row[`Phone ${i + 1} Value`] = p.value;
      });

      // Flatten Emails
      c.emails.forEach((e, i) => {
        row[`Email ${i + 1} Label`] = e.label;
        row[`Email ${i + 1} Value`] = e.value;
      });

      return row;
    });

    const worksheet = XLSX.utils.json_to_sheet(rows);
    const workbook = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(workbook, worksheet, "Contacts");
    XLSX.writeFile(workbook, "My_Contacts.xlsx");
  },

  importContacts: async (file: File, userId: string): Promise<Contact[]> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const data = e.target?.result;
          const workbook = XLSX.read(data, { type: 'binary' });
          const sheetName = workbook.SheetNames[0];
          const sheet = workbook.Sheets[sheetName];
          const json = XLSX.utils.sheet_to_json(sheet);

          const contacts: Contact[] = json.map((row: any) => {
            const phones: ContactMethod[] = [];
            const emails: ContactMethod[] = [];

            // Reconstruct Arrays
            // A primitive way to check for up to 10 potential columns
            for (let i = 1; i <= 5; i++) {
                if (row[`Phone ${i} Value`]) {
                    phones.push({ id: uuidv4(), label: row[`Phone ${i} Label`] || 'Mobile', value: String(row[`Phone ${i} Value`]) });
                }
                if (row[`Email ${i} Value`]) {
                    emails.push({ id: uuidv4(), label: row[`Email ${i} Label`] || 'Work', value: String(row[`Email ${i} Value`]) });
                }
            }

            return {
              id: uuidv4(),
              userId,
              fullName: row['Full Name'] || 'Unknown',
              company: row['Company'],
              jobTitle: row['Job Title'],
              isFavorite: row['Is Favorite'] === 'Yes',
              notes: row['Notes'],
              phones,
              emails,
              addresses: [], // Address parsing logic omitted for brevity in simple Excel
              createdAt: Date.now()
            };
          });

          resolve(contacts);
        } catch (error) {
          reject(error);
        }
      };
      reader.onerror = (err) => reject(err);
      reader.readAsBinaryString(file);
    });
  }
};