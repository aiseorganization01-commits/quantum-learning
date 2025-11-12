// AIMLSE Secure Login System - Admin-Only Backend
// server.js

require('dotenv').config();
const express = require('express');
const session = require('express-session');
const bcrypt = require('bcryptjs');
const { google } = require('googleapis');
const passport = require('passport');
const GoogleStrategy = require('passport-google-oauth20').Strategy;
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3000;

// ===========================
// MIDDLEWARE
// ===========================
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(express.static('public'));

app.use(session({
  secret: process.env.SESSION_SECRET || 'your-super-secret-change-this',
  resave: false,
  saveUninitialized: false,
  cookie: {
    secure: process.env.NODE_ENV === 'production',
    httpOnly: true,
    maxAge: 24 * 60 * 60 * 1000 // 24 hours
  }
}));

app.use(passport.initialize());
app.use(passport.session());

// ===========================
// GOOGLE SHEETS SETUP
// ===========================
const auth = new google.auth.GoogleAuth({
  keyFile: process.env.SERVICE_ACCOUNT_KEY_PATH || './service-account-key.json',
  scopes: ['https://www.googleapis.com/auth/spreadsheets']
});

const sheets = google.sheets({ version: 'v4', auth });
const SPREADSHEET_ID = process.env.SPREADSHEET_ID || '1X4NkrnER-f-uM1gsZOT7KU6X0sIdHZaSoyeKuDrj_Wo';
const SHEET_NAME = 'AIMLSE_Users';

// ===========================
// GOOGLE SHEETS FUNCTIONS
// ===========================

// Get all users from Google Sheets
async function getAllUsers() {
  try {
    const response = await sheets.spreadsheets.values.get({
      spreadsheetId: SPREADSHEET_ID,
      range: `${SHEET_NAME}!A:E`
    });

    const rows = response.data.values;
    if (!rows || rows.length === 0) return [];

    const [headers, ...dataRows] = rows;
    return dataRows.map(row => ({
      name: row[0] || '',
      email: row[1] || '',
      username: row[2] || '',
      password: row[3] || '',
      status: row[4] || 'Student'
    }));
  } catch (error) {
    console.error('Error getting users:', error);
    return [];
  }
}

// Find user by username
async function findUserByUsername(username) {
  const users = await getAllUsers();
  return users.find(u => u.username === username);
}

// Find user by email
async function findUserByEmail(email) {
  const users = await getAllUsers();
  return users.find(u => u.email === email);
}

// Add new user to Google Sheets
async function addUser(name, email, username, password, status = 'Student') {
  try {
    await sheets.spreadsheets.values.append({
      spreadsheetId: SPREADSHEET_ID,
      range: `${SHEET_NAME}!A:E`,
      valueInputOption: 'USER_ENTERED',
      resource: {
        values: [[name, email, username, password, status]]
      }
    });
    return true;
  } catch (error) {
    console.error('Error adding user:', error);
    return false;
  }
}

// Update user in Google Sheets
async function updateUser(oldUsername, newData) {
  try {
    const response = await sheets.spreadsheets.values.get({
      spreadsheetId: SPREADSHEET_ID,
      range: `${SHEET_NAME}!A:E`
    });

    const rows = response.data.values;
    if (!rows || rows.length === 0) return false;

    // Find row index (add 1 because sheets are 1-indexed, add 1 more for header)
    let rowIndex = -1;
    for (let i = 1; i < rows.length; i++) {
      if (rows[i][2] === oldUsername) {
        rowIndex = i + 1; // Convert to 1-indexed
        break;
      }
    }

    if (rowIndex === -1) return false;

    // Update the row
    await sheets.spreadsheets.values.update({
      spreadsheetId: SPREADSHEET_ID,
      range: `${SHEET_NAME}!A${rowIndex}:E${rowIndex}`,
      valueInputOption: 'USER_ENTERED',
      resource: {
        values: [[
          newData.name,
          newData.email,
          newData.username,
          newData.password,
          newData.status
        ]]
      }
    });

    return true;
  } catch (error) {
    console.error('Error updating user:', error);
    return false;
  }
}

// Delete user from Google Sheets
async function deleteUser(username) {
  try {
    const response = await sheets.spreadsheets.values.get({
      spreadsheetId: SPREADSHEET_ID,
      range: `${SHEET_NAME}!A:E`
    });

    const rows = response.data.values;
    if (!rows || rows.length === 0) return false;

    // Find row index
    let rowIndex = -1;
    for (let i = 1; i < rows.length; i++) {
      if (rows[i][2] === username) {
        rowIndex = i;
        break;
      }
    }

    if (rowIndex === -1) return false;

    // Delete the row
    await sheets.spreadsheets.batchUpdate({
      spreadsheetId: SPREADSHEET_ID,
      resource: {
        requests: [{
          deleteDimension: {
            range: {
              sheetId: 0, // Usually the first sheet
              dimension: 'ROWS',
              startIndex: rowIndex,
              endIndex: rowIndex + 1
            }
          }
        }]
      }
    });

    return true;
  } catch (error) {
    console.error('Error deleting user:', error);
    return false;
  }
}

// ===========================
// PASSPORT CONFIGURATION
// ===========================
passport.serializeUser((user, done) => {
  done(null, user.username);
});

passport.deserializeUser(async (username, done) => {
  try {
    const user = await findUserByUsername(username);
    done(null, user);
  } catch (error) {
    done(error);
  }
});

// Google OAuth Strategy
passport.use(new GoogleStrategy({
  clientID: process.env.GOOGLE_CLIENT_ID,
  clientSecret: process.env.GOOGLE_CLIENT_SECRET,
  callbackURL: process.env.CALLBACK_URL || 'http://localhost:3000/auth/google/callback'
}, async (accessToken, refreshToken, profile, done) => {
  try {
    const email = profile.emails[0].value;
    const user = await findUserByEmail(email);

    if (user) {
      // User exists in database
      return done(null, user);
    } else {
      // User not found - Google Sign-In not allowed without admin creating account
      return done(null, false, { message: 'Account not found. Please contact an administrator.' });
    }
  } catch (error) {
    return done(error);
  }
}));

// ===========================
// AUTHENTICATION MIDDLEWARE
// ===========================
function isAuthenticated(req, res, next) {
  if (req.isAuthenticated()) {
    return next();
  }
  res.status(401).json({ error: 'Not authenticated' });
}

function isAdmin(req, res, next) {
  if (req.isAuthenticated() && req.user.status === 'Admin') {
    return next();
  }
  res.status(403).json({ error: 'Admin access required' });
}

// ===========================
// ROUTES - AUTHENTICATION
// ===========================

// Manual Login
app.post('/api/login', async (req, res) => {
  const { username, password } = req.body;

  if (!username || !password) {
    return res.status(400).json({ error: 'Username and password required' });
  }

  try {
    const user = await findUserByUsername(username);

    if (!user) {
      return res.status(401).json({ error: 'Invalid username or password' });
    }

    // Check if this is an OAuth-only account
    if (user.password === 'OAUTH_USER') {
      return res.status(401).json({ error: 'This account uses Google Sign-In only' });
    }

    // Verify password
    const isValid = await bcrypt.compare(password, user.password);
    if (!isValid) {
      return res.status(401).json({ error: 'Invalid username or password' });
    }

    // Login successful
    req.login(user, (err) => {
      if (err) {
        return res.status(500).json({ error: 'Login failed' });
      }
      res.json({ success: true, user: { name: user.name, status: user.status } });
    });

  } catch (error) {
    console.error('Login error:', error);
    res.status(500).json({ error: 'Server error' });
  }
});

// Google OAuth Routes
app.get('/auth/google',
  passport.authenticate('google', { scope: ['profile', 'email'] })
);

app.get('/auth/google/callback',
  passport.authenticate('google', { failureRedirect: '/?error=oauth_failed' }),
  (req, res) => {
    res.redirect('/dashboard.html');
  }
);

// Logout
app.post('/api/logout', (req, res) => {
  req.logout((err) => {
    if (err) {
      return res.status(500).json({ error: 'Logout failed' });
    }
    res.json({ success: true });
  });
});

// Get current user
app.get('/api/user', isAuthenticated, (req, res) => {
  res.json({
    name: req.user.name,
    email: req.user.email,
    username: req.user.username,
    status: req.user.status
  });
});

// ===========================
// ROUTES - DASHBOARD
// ===========================
app.get('/api/dashboard', isAuthenticated, async (req, res) => {
  const user = req.user;
  
  let content = {
    features: [],
    lessons: []
  };

  // Customize based on role
  if (user.status === 'Student') {
    content.features = [
      'AI Programming',
      'Machine Learning',
      'Quantum Computing',
      'Python Coding',
      'Team Projects',
      'Discord Community'
    ];
    content.lessons = [
      { title: 'Introduction to Machine Learning', completed: true },
      { title: 'Python Fundamentals', completed: true },
      { title: 'Neural Networks Basics', completed: false },
      { title: 'Quantum Computing Intro', completed: false }
    ];
  } else if (user.status === 'Teacher') {
    content.features = [
      'Student Progress',
      'Lesson Management',
      'Class Analytics',
      'Resource Library'
    ];
    content.lessons = [
      { title: 'Class A - Period 1', students: 24 },
      { title: 'Class B - Period 3', students: 22 },
      { title: 'Class C - Period 5', students: 26 }
    ];
  } else if (user.status === 'Admin') {
    content.features = [
      'User Management',
      'System Analytics',
      'Content Management',
      'Security Settings'
    ];
    content.lessons = [
      { title: 'Total Users', students: 127 },
      { title: 'Active Classes', students: 8 },
      { title: 'Completed Lessons', students: 456 }
    ];
  }

  res.json({ user: { name: user.name, status: user.status }, content });
});

// ===========================
// ROUTES - ADMIN ONLY
// ===========================

// Get all users (Admin only)
app.get('/api/admin/users', isAdmin, async (req, res) => {
  try {
    const users = await getAllUsers();
    // Don't send password hashes to frontend
    const safeUsers = users.map(u => ({
      name: u.name,
      email: u.email,
      username: u.username,
      status: u.status,
      isOAuth: u.password === 'OAUTH_USER'
    }));
    res.json({ users: safeUsers });
  } catch (error) {
    res.status(500).json({ error: 'Failed to get users' });
  }
});

// Add new user (Admin only)
app.post('/api/admin/users', isAdmin, async (req, res) => {
  const { name, email, username, password, status, isOAuth } = req.body;

  if (!name || !email || !username || !status) {
    return res.status(400).json({ error: 'All fields required' });
  }

  if (!isOAuth && !password) {
    return res.status(400).json({ error: 'Password required for manual login accounts' });
  }

  try {
    // Check if username or email already exists
    const existingUsername = await findUserByUsername(username);
    const existingEmail = await findUserByEmail(email);

    if (existingUsername) {
      return res.status(400).json({ error: 'Username already exists' });
    }

    if (existingEmail) {
      return res.status(400).json({ error: 'Email already exists' });
    }

    // Hash password or mark as OAuth
    let finalPassword;
    if (isOAuth) {
      finalPassword = 'OAUTH_USER';
    } else {
      finalPassword = await bcrypt.hash(password, 10);
    }

    // Add user to sheet
    const success = await addUser(name, email, username, finalPassword, status);

    if (success) {
      res.json({ success: true, message: 'User created successfully' });
    } else {
      res.status(500).json({ error: 'Failed to create user' });
    }
  } catch (error) {
    console.error('Error creating user:', error);
    res.status(500).json({ error: 'Server error' });
  }
});

// Update user (Admin only)
app.put('/api/admin/users/:username', isAdmin, async (req, res) => {
  const oldUsername = req.params.username;
  const { name, email, username, password, status, isOAuth } = req.body;

  if (!name || !email || !username || !status) {
    return res.status(400).json({ error: 'All fields required' });
  }

  try {
    // Check if new username/email conflicts with other users
    if (username !== oldUsername) {
      const existingUsername = await findUserByUsername(username);
      if (existingUsername) {
        return res.status(400).json({ error: 'Username already exists' });
      }
    }

    const currentUser = await findUserByUsername(oldUsername);
    if (email !== currentUser.email) {
      const existingEmail = await findUserByEmail(email);
      if (existingEmail) {
        return res.status(400).json({ error: 'Email already exists' });
      }
    }

    // Determine final password
    let finalPassword;
    if (isOAuth) {
      finalPassword = 'OAUTH_USER';
    } else if (password && password.trim() !== '') {
      // New password provided
      finalPassword = await bcrypt.hash(password, 10);
    } else {
      // Keep existing password
      finalPassword = currentUser.password;
    }

    // Update user
    const success = await updateUser(oldUsername, {
      name,
      email,
      username,
      password: finalPassword,
      status
    });

    if (success) {
      res.json({ success: true, message: 'User updated successfully' });
    } else {
      res.status(500).json({ error: 'Failed to update user' });
    }
  } catch (error) {
    console.error('Error updating user:', error);
    res.status(500).json({ error: 'Server error' });
  }
});

// Delete user (Admin only)
app.delete('/api/admin/users/:username', isAdmin, async (req, res) => {
  const username = req.params.username;

  // Prevent deleting yourself
  if (username === req.user.username) {
    return res.status(400).json({ error: 'Cannot delete your own account' });
  }

  try {
    const success = await deleteUser(username);

    if (success) {
      res.json({ success: true, message: 'User deleted successfully' });
    } else {
      res.status(500).json({ error: 'Failed to delete user' });
    }
  } catch (error) {
    console.error('Error deleting user:', error);
    res.status(500).json({ error: 'Server error' });
  }
});

// ===========================
// START SERVER
// ===========================
app.listen(PORT, () => {
  console.log(`\n🚀 AIMLSE Login System running on http://localhost:${PORT}`);
  console.log(`📊 Google Sheet ID: ${SPREADSHEET_ID}`);
  console.log(`🔐 Google OAuth callback: ${process.env.CALLBACK_URL || 'http://localhost:3000/auth/google/callback'}`);
  console.log(`\n✅ Admin-only user management enabled`);
  console.log(`👤 Default admin setup required in Google Sheets\n`);
});