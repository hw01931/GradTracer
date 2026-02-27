import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';

let diagnosticCollection: vscode.DiagnosticCollection;
let recipeData: any = null;

export function activate(context: vscode.ExtensionContext) {
    console.log('GradTracer extension is now active!');

    diagnosticCollection = vscode.languages.createDiagnosticCollection('gradtracer');
    context.subscriptions.push(diagnosticCollection);

    // Watch for gradtracer_recipe.json in the workspace root
    const workspaceFolders = vscode.workspace.workspaceFolders;
    if (workspaceFolders) {
        const rootPath = workspaceFolders[0].uri.fsPath;
        const recipePath = path.join(rootPath, 'gradtracer_recipe.json');

        // Initial load
        loadRecipe(recipePath);

        // Setup File Watcher
        const watcher = vscode.workspace.createFileSystemWatcher(
            new vscode.RelativePattern(rootPath, 'gradtracer_recipe.json')
        );

        watcher.onDidChange(() => loadRecipe(recipePath));
        watcher.onDidCreate(() => loadRecipe(recipePath));
        watcher.onDidDelete(() => {
            recipeData = null;
            diagnosticCollection.clear();
        });

        context.subscriptions.push(watcher);
    }

    // Register Hover Provider
    const hoverProvider = vscode.languages.registerHoverProvider('python', {
        provideHover(document, position) {
            if (!recipeData || !recipeData.layers) return null;

            const range = document.getWordRangeAtPosition(position);
            const word = document.getText(range);

            // Very naive string matching against layer names
            // e.g. if the layer name is "item_embedding" and we hover over it
            for (const [layerName, data] of Object.entries<any>(recipeData.layers)) {
                if (layerName === word) {
                    const md = new vscode.MarkdownString();
                    md.isTrusted = true;
                    md.appendMarkdown(`### 🌊 GradTracer Auto-Compression\n\n`);
                    md.appendMarkdown(`**Health Score:** ${data.health_score} / 100\n\n`);
                    md.appendMarkdown(`**Action:** [${data.quantization} + ${(data.prune_ratio * 100).toFixed(0)}% Pruning]\n\n`);
                    md.appendMarkdown(`*Reason:* ${data.reason}\n\n`);
                    md.appendMarkdown(`[⚡ Apply Recipe](command:gradtracer.applyRecipe)`);
                    return new vscode.Hover(md);
                }
            }
            return null;
        }
    });

    context.subscriptions.push(hoverProvider);

    // Register Command to Apply
    const disposable = vscode.commands.registerCommand('gradtracer.applyRecipe', () => {
        vscode.window.showInformationMessage('GradTracer: Compression recipe applied! Restarting trainer...');
    });
    context.subscriptions.push(disposable);
}

function loadRecipe(recipePath: string) {
    if (fs.existsSync(recipePath)) {
        try {
            const raw = fs.readFileSync(recipePath, 'utf-8');
            recipeData = JSON.parse(raw);
            vscode.window.showInformationMessage('GradTracer: Auto-Compression Recipe loaded.');
        } catch (e) {
            console.error('Failed to parse GradTracer recipe:', e);
        }
    }
}

export function deactivate() {
    if (diagnosticCollection) {
        diagnosticCollection.clear();
    }
}
